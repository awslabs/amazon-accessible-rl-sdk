# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import annotations

import os
import warnings
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from random import randrange
from time import perf_counter
from typing import Callable, Generic, TypeVar, Union, cast

import gym
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from loguru import logger
from matplotlib.pyplot import Axes
from nptyping import Integer, NDArray, Shape
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from torch.utils.data import Dataset
from typing_extensions import TypeGuard

import a2rl as wi
from a2rl import WiDataFrame
from a2rl.mingpt.model import GPT, GPTConfig
from a2rl.mingpt.trainer import Trainer, TrainerConfig
from a2rl.tokenizer import DiscreteTokenizer
from a2rl.utils import pickle_save

TRAIN_CONFIG_KEYS = [
    "epochs",
    "batch_size",
    "embedding_dim",
    "gpt_n_layer",
    "gpt_n_head",
    "learning_rate",
    "num_workers",
    "lr_decay",
]


def is_npinstance(o, t) -> bool:
    """Wrapper to nptyping.isinstance() to silence mypy complaints about parameterized generics."""
    return isinstance(o, t)


def model_forward(model: Callable, *args, **kwargs) -> torch.Tensor:
    """This is a hack to get the logits from either mingpt.forward() which returning (logits, loss),
    or lightgpt.forward() which returns logits."""
    result: tuple[torch.Tensor, ...] = model(*args, **kwargs)
    if isinstance(result, tuple):
        return result[0]  # mingpt (plain torch) .forward() returns (logits, loss).
    return result  # lightgpt .forward() returns logits.


@dataclass
class SimulatorDataset(Dataset):
    """Transform a 1D numpy array into PyTorch dataset.

    Arguments:
        sequence: Input numpy array of tokenized dataframe values.
        block_size: Context length.

    Examples
    --------
    To create new simulator dataset.

    .. code-block:: python

        >>> from a2rl.simulator import SimulatorDataset
        >>> import numpy as np

        >>> input_seq = np.array([0, 1, 10, 11])
        >>> block_size = 2
        >>> ds = SimulatorDataset(input_seq, block_size)
        >>> ds[0]
        (tensor([0, 1]), tensor([1, 2]))

    """

    sequence: np.ndarray = field(repr=False)
    block_size: int

    def __post_init__(self):
        if not is_npinstance(self.sequence, NDArray[Shape["*"], Integer]):
            raise TypeError(
                f"Expect sequence as a 1D int array, but got this instead: {repr(self.sequence)}"
            )

        chars = sorted(list(set(self.sequence)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data_size, self.vocab_size = len(self.sequence), len(chars)

    def __len__(self):
        return len(self.sequence) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.sequence[idx : idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


@dataclass
class AutoTokenizer:
    """Auto tokenizer process input Whatif dataset and provide data-level helper functions for
    Trainer and Simulator.

    ``Dataframe token`` refers to the tokenized dataframe column values.
    ``GPT token`` refers to the input token passed to GPT model.

    ``tokenized_val_to_gpt_token_map`` property give the mapping between dataframe
    token and GPT token.

    Arguments:
        df: This is a WiDataFrame.

        block_size_row: Number of rows to be used as context windows for GPT model.
                        If there are ``n`` columns in dataframe, the context windows is
                        calculated as ``n * block_size_row`` tokens.

        train_ratio: The ratio of data to be used for training. Default is 0.8 (80%).

    .. note::
        Context length that is greater than ``block_size_row`` will be discarded
        before passing to GPT model for next token prediction.

    Examples
    --------
    You can instantiate a `AutoTokenizer` with `whatif` dataframe, and
    specific the `block_size_row` in term of number of rows in dataframe.

    .. code-block:: python

        >>> import a2rl as wi
        >>> import numpy as np
        >>> import pandas as pd

        >>> df = pd.DataFrame(
        ...     np.array(
        ...         [
        ...             [0, 10, 20, 200],
        ...             [1, 12, 21, 225],
        ...             [2, 15, 22, 237],
        ...         ]
        ...     ),
        ...     columns=["s1", "s2", "a", "r"],
        ... )
        >>> wi_df = wi.WiDataFrame(df, states=["s1", "s2"], actions=["a"], rewards=["r"])
        >>> wi_df.add_value()
           s1  s2   a    r       value
        0   0  10  20  200  122.610372
        1   1  12  21  225  105.776159
        2   2  15  22  237    0.000000

    Retrived discretized dataframe using ``df_tokenized`` property.

    .. code-block:: python

        >>> field_tokenizer = wi.DiscreteTokenizer(num_bins_strategy="uniform")
        >>> tokenizer = wi.AutoTokenizer(wi_df, 1, field_tokenizer=field_tokenizer)
        >>> tokenizer.df_tokenized
           s1   s2    a    r  value
        0   0  100  200  300    499
        1  50  140  250  367    486
        2  99  199  299  399    400


    To tokenize a new dataframe, use :meth:`AutoTokenizer.field_tokenizer.transform`

    .. code-block:: python

        >>> new_df = pd.DataFrame(
        ...     np.array(
        ...         [
        ...             [0, 14, 25, 210],
        ...             [2, 15, 26, 211],
        ...         ]
        ...     ),
        ...     columns=["s1", "s2", "a", "r"],
        ... )
        >>> new_wi_df = wi.WiDataFrame(new_df, states=["s1", "s2"], actions=["a"], rewards=["r"])
        >>> new_wi_df = new_wi_df.add_value()
        >>> tokenizer.field_tokenizer.transform(new_wi_df)
           s1   s2    a    r  value
        0   0  180  299  327    476
        1  99  199  299  329    400

    .. note::
        The data for each column cannot have just a single value.

        In order to reuse a tokenizer, the dataframe must have the same columns.
        In this example, you must create the ``value`` column as well by calling ``add_value()``.


    You can transform the dataframe token into GPT token or vice
    versa as follows.

    .. code-block:: python

        >>> seq = np.array([0, 100, 200, 300, 499])
        >>> gpt_token = tokenizer.gpt_tokenize(seq)
        >>> gpt_token
        array([ 0,  3,  6,  9, 14])

        >>> gpt_token_inv = tokenizer.gpt_inverse_tokenize(gpt_token)
        >>> gpt_token_inv
        array([  0, 100, 200, 300, 499])

    To convert sequence back into dataframe.

    .. code-block:: python

        >>> tokenizer.from_seq_to_dataframe(seq) # doctest:+SKIP
             s1      s2      a        r      value
        0  0.01  10.025  20.01  200.185  121.99732

    """

    df: WiDataFrame = field(init=True, repr=False)
    block_size_row: int = field(init=True, repr=False)
    train_ratio: float = 1.0
    copy: bool = field(default=True, repr=False)
    field_tokenizer: DiscreteTokenizer = field(default_factory=DiscreteTokenizer, repr=False)

    def __post_init__(self):

        self.df = self.df.trim(self.copy)
        self.columns = self.df.columns
        self.column_len = len(self.columns)
        self.state_columns = self.df.states
        self.action_columns = self.df.actions
        self.reward_columns = self.df.rewards
        self.df_shape: tuple = self.df.shape
        self.field_tokenizer = self.field_tokenizer.fit(self.df)
        self.df_tokenized = self.field_tokenizer.transform(self.df)
        self.state_indices: np.ndarray = self.df.columns.get_indexer(self.state_columns)
        self.action_indices: np.ndarray = self.df.columns.get_indexer(self.action_columns)
        self.reward_indices: np.ndarray = self.df.columns.get_indexer(self.reward_columns)
        self.state_dim = len(self.state_indices)
        self.action_dim = len(self.action_indices)
        self.reward_dim = len(self.reward_indices)
        self.block_size = self.block_size_row * self.column_len
        self.sequence_len = len(self.df_tokenized.sequence)
        if self.sequence_len - self.block_size <= 0:
            raise ValueError(
                f"Dataframe has {self.df_shape[0]} rows and {self.df_shape[1]} columns "
                f"with a total sequence length of {self.sequence_len}, "
                f"but block_size_row of {self.block_size_row} requires a sequence length of "
                f"{self.block_size}, which is greater than the max of {self.sequence_len}. "
                "Try to reduce block_size_row."
            )

        self._get_gym_enc_to_action()
        self._get_col_eligible_index()

        self.simulator_ds = SimulatorDataset(
            sequence=self.df_tokenized.sequence, block_size=self.block_size
        )
        self.vocab_size = self.simulator_ds.vocab_size
        self._gpt_token_to_tokenized_val_map = self.simulator_ds.itos
        self._tokenized_val_to_gpt_token_map = self.simulator_ds.stoi
        known_df_token = np.array(list(self._gpt_token_to_tokenized_val_map.values())).reshape(
            -1, 1
        )
        self.token_neighbors = NearestNeighbors(n_neighbors=1).fit(known_df_token)

        train_size = int(len(self.simulator_ds) * self.train_ratio)
        test_size = len(self.simulator_ds) - train_size
        self.train_dataset = torch.utils.data.Subset(self.simulator_ds, range(train_size))
        self.test_dataset = torch.utils.data.Subset(
            self.simulator_ds, range(train_size, train_size + test_size)
        )

    def _get_gym_enc_to_action(self):
        """Create the mapping between gym encoded action and action string."""

        self._gym_enc_to_action, self._gym_action_to_enc = {}, {}
        for i, col_name in enumerate(self.action_columns):
            action_str = self.df[self.action_columns[i]].unique()
            action_str.sort()
            self._gym_enc_to_action[col_name] = {i: a for i, a in enumerate(action_str)}
            self._gym_action_to_enc[col_name] = {a: i for i, a in enumerate(action_str)}

    def _get_col_eligible_index(self):
        """Create a dict of eligible tokenized value range for each column in whatif
        dataframe.

        The dict mapping has the format of `{col1_index: [min, max), col2_index: [min, max)}`

        col_eligible_index = {0: [0, 100), 1: [100, 200)}

        Column 0 has a value range between 0 and 99 inclusive of both boundary.

        """
        self._col_eligible_index: dict[int, list] = {}
        for i in range(len(self.columns)):
            valid_tokens = self.field_tokenizer.valid_tokens(i)
            self._col_eligible_index[i] = [min(valid_tokens), max(valid_tokens) + 1]

    @property
    def col_eligible_index(self):
        return self._col_eligible_index

    @property
    def gym_enc_to_action(self):
        return self._gym_enc_to_action

    @property
    def gym_action_to_enc(self):
        return self._gym_action_to_enc

    @property
    def gpt_token_to_tokenized_val_map(self):
        return self._gpt_token_to_tokenized_val_map

    @property
    def tokenized_val_to_gpt_token_map(self):
        return self._tokenized_val_to_gpt_token_map

    def gpt_tokenize(self, seq: np.ndarray) -> np.ndarray:
        """Convert input sequence from dataframe token to GPT token."""

        if not isinstance(seq, np.ndarray):
            raise TypeError(f"seq must be a numpy array, but got {type(seq)}")

        key_unique = self.tokenized_val_to_gpt_token_map.keys()
        mask = np.array([i in key_unique for i in seq])

        check_all_exists = np.all(mask)
        if not check_all_exists:
            raise ValueError(
                f"There is dataframe token {seq[~mask]} that does not exist in whatif "
                "dataframe token used to instantiate Autotokenizer. You can find the valid "
                "dataframe token in Tokenizer.tokenized_val_to_gpt_token_map.keys()"
            )

        x = np.array([self.tokenized_val_to_gpt_token_map[s] for s in seq])
        return x

    def gpt_inverse_tokenize(self, seq: np.ndarray) -> np.ndarray:
        """Convert input sequence from GPT token to dataframe token."""

        if not isinstance(seq, np.ndarray):
            raise TypeError(f"seq must be a numpy array, but got {type(seq)}")

        key_unique = self.gpt_token_to_tokenized_val_map.keys()
        mask = np.array([i in key_unique for i in seq])
        check_all_exists = np.all(mask)
        if not check_all_exists:
            raise ValueError(
                f"There is GPT token {seq[~mask]} that does not exist in whatif GPT token "
                "used to instantiate Autotokenizer. You can find the valid "
                "GPT token in Tokenizer.gpt_token_to_tokenized_val_map.keys()"
            )

        x = np.array([self.gpt_token_to_tokenized_val_map[s] for s in seq])
        return x

    def from_seq_to_dataframe(
        self,
        seq: np.ndarray,
        inverse: bool = True,
    ) -> pd.DataFrame:
        """Convert sequence of tokenized value back into original value, in the
        form of dataframe.

        Arguments:
            seq: The sequence length must be of multiple of column lenght.
            inverse: Converted dataframe token back into original value when True,
                     else it still stay as dataframe token when False.

        Returns:
            DataFrame of original values.
        """
        if not isinstance(seq, np.ndarray):
            raise TypeError(f"seq must be a numpy array, but got {type(seq)}")

        if len(seq) % self.column_len != 0:
            raise ValueError(
                f"seq length must be multiple of column length {self.column_len} "
                f"in order to fit into dataframe column, but got {len(seq)}",
            )

        seq = seq.reshape(-1, self.column_len)
        df = WiDataFrame(
            data=seq,
            columns=self.df.columns,
            **self.df.sar_d,
        )
        if not inverse:
            return df
        return self.field_tokenizer.inverse_transform(df)


Model_T = TypeVar("Model_T")
Trainer_T = TypeVar("Trainer_T")


@dataclass
class BaseBuilder(ABC, Generic[Model_T, Trainer_T]):
    """Provides high-level APIs for training and evaluating a model using :class:`AutoTokenizer`
    data, shielding callers from low-level constructs such as the underlying PyTorch module,
    trainer object, etc.

    It has no knowledge of dataframe shape, and which values belong to action/states/reward.
    """

    tokenizer: AutoTokenizer
    model_dir: str | Path | None = None
    config: dict | str | Path | None = None

    # Concrete builders who don't manually place tensor to devices MUST implement a __post_init__()
    # that behaves as follow:
    #
    # class SampleConcreteBuilder(BaseBuilder):
    #     def __post_init(self):
    #         self.manage_tensor_placement = False
    #         super().__post_init__()
    #         ...  # Stuffs specific to this subclass.
    #
    # This posture implies that we concede most subclass won't venture to the land of
    # pytorch-lightning or huggingface-accelerate (with automatic device placement).
    manage_tensor_placement: bool = field(default=True, init=False)

    # Hack for python<3.10: due to the absence of dataclass(kw_only=True) whici is available in
    # Python 3.10+, subclass uses this rather funny dictionary for additional "kwargs" in their
    # __init__().
    kw_args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.model_dir is None:
            utc_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            salt = randrange(0x7FFFFFFF)  # nosec B311 -- this is not for cryptographic purpose.
            self.model_dir = Path(f"model-a2rl-{utc_ts}utc-{salt}")

        self.model_name = "model.pt"

        self.loaded_config = self.config_loader()
        self._check_config()
        self._configs = self.loaded_config["train_config"]
        self._fitted = False

        if self.manage_tensor_placement:
            # Default implementation is barebone, and recognizes cpu and cuda only.
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = torch.cuda.current_device()  # type: ignore
        else:
            self.device = None
        # print(f"{self.device=}")

    def sample(
        self,
        seq: np.ndarray,
        n_steps: int,
        temperature: float = 1.0,
        sample: bool = False,
        top_k: bool = False,
    ) -> np.ndarray:
        """Sample the next ``n_steps`` token.

        Arguments:
            seq: These is a sequence of GPT tokens. You need to convert dataframe token to GPT token
                using ``Tokenizer.gpt_tokenize()``
            n_steps: Number of steps to predict.
            temperature: The temperature controls the randomness of predicted samples by scaling the
                logits before applying softmax.
            sample: When ``True``, returns random samples of actions from the ``top-k`` logits.
                Otherwise, straightaway returns the ``top-k`` logits.
            top_k: The number of logits to consider for the returned actions.

        Returns:
            The original context, concatenated with the next ``n_steps`` predicted token.
        """
        if not self._fitted:
            raise ValueError("Please make sure fit() or load_model() has been called.")

        if not isinstance(seq, np.ndarray):
            raise TypeError(f"seq must be a numpy array, but got {type(seq)}")

        if seq.ndim != 1:
            raise ValueError(f"seq shape must have dim of 1, but got {seq.ndim}")

        # Make sure the correct trained model is used.
        if seq.max() > self._model.tok_emb.num_embeddings:
            raise ValueError(
                "The model has not seen the seq dataset. "
                f"Max num of embedding {self._model.tok_emb.num_embeddings} is smaller than "
                f"input token value of {seq.max()}",
            )

        x = torch.tensor(seq, dtype=torch.long)[None, ...]
        if self.manage_tensor_placement:
            x = x.to(self.device)

        block_size = self._model.get_block_size()  # max context size 128
        self._model.eval()
        for k in range(n_steps):
            # Crop context window if needed
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits = model_forward(self._model, x_cond)
            # Get the next token
            logits = logits[:, -1, :] / temperature

            if top_k:
                logits = top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                # Return: (value, idx), e.g. [0.1,0.2,0.7]: (0.7,2)
                _, ix = torch.topk(probs, k=1, dim=-1)

            # Append to the sequence and continue
            x = torch.cat((x, ix), dim=1)

        if self.manage_tensor_placement and self.device != "cpu":
            x = x.cpu()
        x_np = x.numpy().flatten()
        return x_np

    def evaluate(self, context_len: int = 20, sample: bool = True, horizon: int = None) -> Axes:
        """This is to evaluate the raw GPT model.

        Arguments:
            context_len: These is a sequence of GPT tokens
            sample: Enable sampling.
            horizon: The number of GPT token to predict based on input GPT token sequence.

        Returns:
            Matplotlib Axes.
        """

        if horizon is None:
            horizon = 200

        test_seq = self.tokenizer.df_tokenized.sequence[:context_len]
        test_gpt_token = self.tokenizer.gpt_tokenize(test_seq)
        preds_gpt = self.sample(test_gpt_token, n_steps=(horizon), sample=sample)
        preds_seq = self.tokenizer.gpt_inverse_tokenize(preds_gpt)

        true_ser = pd.Series(
            self.tokenizer.df_tokenized.sequence[: horizon + context_len], name="true"
        )
        pred_ser = pd.Series(preds_seq, name="pred")
        true_df = pd.concat([true_ser, pred_ser], axis=1)
        styles = ["-o", "-x"]
        title = (
            f"Actual vs Prediction in GPT token space.\n"
            f"context_len={context_len}, sample={sample}, horizon={horizon}"
        )
        ax = true_df.plot(
            style=styles,
            figsize=(15, 5),
            title=title,
        )
        ax.axvline(x=context_len, color="red")
        ax.text(
            x=0.03,
            y=0.05,
            s="Context\nwindow",
            fontsize=15,
            transform=ax.transAxes,
            fontweight="bold",
            c="green",
            alpha=0.5,
        )
        ax.legend(loc="lower right")
        return ax

    def _check_config(self):
        """Check config has the required keys."""

        if "train_config" not in self.loaded_config.keys():
            raise ValueError('Invalid config. Missing key "train_config"')

        if not all(i in self.loaded_config["train_config"] for i in TRAIN_CONFIG_KEYS):
            raise ValueError(f"Invalid config. Missing one of {TRAIN_CONFIG_KEYS}")

    def config_loader(self) -> dict:
        """Load training configuration.

        Returns:
            Model configuration.
        """
        if isinstance(self.config, dict):
            return deepcopy(self.config)

        if self.config is None:
            self.config = Path(__file__).parent / "config.yaml"
        elif not isinstance(self.config, Path):
            self.config = Path(self.config)

        if not self.config.is_file():
            raise FileNotFoundError(f'Config file "{self.config}" not found.')

        with open(self.config) as stream:
            return yaml.safe_load(stream)

    def save_model(self):
        """Save trained pytorch model, training config, and associated tokenizer.

        Tokenizer and training config will be saved as "tokenizer.pt" and "config.yaml"
        respectively.

        """

        p = Path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, Path(self.model_dir) / self.model_name)

        tokenizer_path = Path(self.model_dir) / "tokenizer.pt"
        pickle_save(tokenizer_path, self.tokenizer)

        config_path = Path(self.model_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(self.loaded_config, f)

    def load_model(self) -> GPT:
        """Load a trained model."""
        model_path = Path(cast(Union[str, Path], self.model_dir)) / self.model_name

        if not model_path.is_file():
            cur_wd = os.getcwd()
            raise FileNotFoundError(
                f'Model file "{model_path}" not found w.r.t current working dir is {cur_wd}.'
            )

        if self.device == "cpu":
            model = torch.load(model_path, map_location="cpu")
        else:
            model = torch.load(model_path)

        self._model = model
        self._fitted = True
        return model

    @property
    def model(self):
        if not hasattr(self, "_model"):
            raise ValueError("Please load the model by calling fit() or load_model() first.")

        return self._model


class GPTBuilder(BaseBuilder[GPT, Trainer]):
    """Provides high-level APIs to train and evaluate a GPT model based on the data loaded in
    :class:`AutoTokenizer`.

    It has no knowledge of dataframe shape, and which values belong to action/states/reward.

    Arguments:
        tokenizer: This is a :class:`AutoTokenizer`.
        model_dir: Model directory for saving and loading. When set to ``None``, automatically
            generate the directory name.
        config: Custom configuration file or dictionary. When set to ``None``, use the built-in
            configuration in ``a2rl/config.yaml``.

    .. note::
        For configuration, precedence start with ``config`` as parameter, followed by
        custom file indicated by ``config_dir`` and ``config_name``.

        If none are specified, default configuration located in ``src/a2rl/config.yaml``
        will be used.

    Configuration file must meet the following ``yaml`` format.

    .. code-block:: yaml

        train_config:
            epochs: 5
            batch_size: 512
            embedding_dim: 512
            gpt_n_layer: 1
            gpt_n_head: 1
            learning_rate: 6e-4
            num_workers: 0
            lr_decay: True

    Examples
    --------
    Train a model, and save to a temporary directory.

    .. code-block:: python

        >>> import tempfile
        >>> import a2rl as wi
        >>> from a2rl.simulator import AutoTokenizer, GPTBuilder

        >>> wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller"))
        >>> tokenizer = AutoTokenizer(wi_df, block_size_row=2)
        >>> with tempfile.TemporaryDirectory() as model_dir:
        ...     builder = GPTBuilder(tokenizer, model_dir)
        ...     model = builder.fit()  # doctest:+SKIP

    Load a pretrained model.

    .. code-block:: python

        >>> wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller"))
        >>> tokenizer = AutoTokenizer(wi_df, block_size_row=2)
        >>> with tempfile.TemporaryDirectory() as model_dir:
        ...     builder = GPTBuilder(tokenizer, model_dir)
        ...     model = builder.fit()  # doctest:+SKIP
        ...     model = builder.load_model() # doctest:+SKIP

    Pass in a custom configuration via parameter.

    .. code-block:: python

        >>> custom_config = {
        ...     "train_config": {
        ...         "epochs": 1,
        ...         "batch_size": 512,
        ...         "embedding_dim": 512,
        ...         "gpt_n_layer": 1,
        ...         "gpt_n_head": 1,
        ...         "learning_rate": 0.0006,
        ...         "num_workers": 0,
        ...         "lr_decay": True,
        ...     }
        ... }
        >>> wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller"))
        >>> tokenizer = AutoTokenizer(wi_df, block_size_row=2)
        >>> with tempfile.TemporaryDirectory() as model_dir:
        ...     builder = GPTBuilder(tokenizer, model_dir, custom_config)
        ...     model = builder.fit()  # doctest:+SKIP
    """

    def fit(self, validate: bool = True) -> GPT:
        """Start training model."""

        logger.info(self._configs)

        # Only expose commonly use configuration in config file.
        mconf = GPTConfig(
            self.tokenizer.vocab_size,
            self.tokenizer.block_size,
            n_layer=self._configs["gpt_n_layer"],
            n_head=self._configs["gpt_n_head"],
            n_embd=self._configs["embedding_dim"],
        )
        self._model = GPT(mconf)

        self.tconf = TrainerConfig(
            max_epochs=self._configs["epochs"],
            batch_size=self._configs["batch_size"],
            learning_rate=float(self._configs["learning_rate"]),
            lr_decay=self._configs["lr_decay"],
            warmup_tokens=512 * 20,  # Use linear warm up for first batch of token (512x20)
            final_tokens=2
            * len(self.tokenizer.train_dataset)
            * self.tokenizer.block_size,  # Use cosine decay after that
            num_workers=self._configs["num_workers"],
        )

        self.trainer = Trainer(
            self._model,
            self.tokenizer.train_dataset,
            self.tokenizer.test_dataset if validate and len(self.tokenizer.test_dataset) else None,
            self.tconf,
        )

        t1_start = perf_counter()
        self.trainer.train()
        t1_stop = perf_counter()
        logger.info(f"Training time in mins: {(t1_stop - t1_start)/60:.02}")
        self.save_model()
        self._fitted = True
        return self.model


@dataclass
class Simulator(gym.Env[np.ndarray, list]):
    """This is a Simulator class that can provide recommendation for an action, and the
    associated value, given the current context.

    The simulator is to be used together with the :class:`Tokenizer` and :class:`GPTBuilder`
    trained model during instantiation.

    Arguments:
        tokenizer: ``AutoTokenizer`` instance.
        model: Trained model from ``GPTBuilder``
        max_steps: Number of steps per episode.
        reset_coldstart: Number of dataframe context rows.
        test_mode: When True, reset current rows to dataframe index zero.

    Examples
    --------
    This example show how to get a recommendation using a simple dataset.

    First by loading the data and generate value column. Refer to :class:`WiDataFrame`.

    .. code-block:: python

        >>> import numpy as np
        >>> import pandas as pd
        >>> import a2rl as wi
        >>>
        >>> df = pd.DataFrame(
        ...     np.array(
        ...         [
        ...             [0, 10, 20, 200],
        ...             [1, 12, 21, 225],
        ...             [2, 15, 22, 237],
        ...         ]
        ...     ),
        ...     columns=["s1", "s2", "a", "r"],
        ... )
        >>> wi_df = wi.WiDataFrame(df, states=["s1", "s2"], actions=["a"], rewards=["r"])
        >>> wi_df.add_value()
           s1  s2   a    r       value
        0   0  10  20  200  122.610372
        1   1  12  21  225  105.776159
        2   2  15  22  237    0.000000

    Next create a :class:`AutoTokenizer` using the dataframe, indicating the desired block size
    in term of number of rows. You can get discretized dataframe token
    via :class:`AutoTokenizer` properties.

    .. code-block:: python

        >>> field_tokenizer = wi.DiscreteTokenizer(num_bins_strategy="uniform")
        >>> tokenizer = wi.AutoTokenizer(wi_df, block_size_row=1, field_tokenizer=field_tokenizer)
        >>> tokenizer.df_tokenized
           s1   s2    a    r  value
        0   0  100  200  300    499
        1  50  140  250  367    486
        2  99  199  299  399    400

    Train a GPT model using :class:GPTBuilder by passing in the :class:`AutoTokenizer`, and
    ``model_dir`` and ``model_name``.

    .. code-block:: python

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as model_dir:
        ...     builder = wi.GPTBuilder(tokenizer, model_dir)
        ...     builder.fit() # doctest:+ELLIPSIS
        GPT(...)

    Get a recommendation by giving a context, and perform ``max_size`` number of sampling.

    .. note::
        The context is in **row major order**, MUST be in the format
        of ``(s,a,r,...,s)`` ending with states, in **discretized dataframe tokens**.

        In this example, the context ``[0, 100, 200, 300, 499, 50, 140]``
        represents ``[s1, s2, a, r, value, s1, s2]``

    .. code-block:: python

        >>> simulator = wi.Simulator(tokenizer, builder.model)
        >>> custom_context = np.array([0,100,200,300,499,50,140])
        >>> rec_df = simulator.sample(custom_context, max_size=2)

    And finally pick an action that corresponding to the minimum or maximum of value column
    depending on your objective.

    .. code-block:: python

        >>> rec_df # doctest:+SKIP
               a        r       value
        0  21.01  224.975  106.057972
        1  21.01  224.975  106.057972

    """

    tokenizer: AutoTokenizer
    model: GPT = field(repr=False)
    max_steps: int = 100
    reset_coldstart: int = 2
    test_mode: bool = True

    def __post_init__(self):
        self.reset()
        self._setup_gym_attributes()

        # HACK: auto-detect whether to manually place tensors or not.
        if isinstance(self.model, pl.LightningModule):
            self.manage_tensor_placement = False
            self.device = None
        else:
            self.manage_tensor_placement = True
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = torch.cuda.current_device()  # type: ignore

    def _setup_gym_attributes(self):
        # Loop through the number of action, then check the number of choices for each action
        action_count_list = []
        for i in self.tokenizer._gym_enc_to_action:
            action_list = self.tokenizer._gym_enc_to_action[i]
            action_count_list.append(len(action_list))
        self.action_space = MultiDiscrete(action_count_list)

        spaces = {}
        for col in self.tokenizer.df.states:
            if is_numeric_dtype(self.tokenizer.df[col]):
                spaces[col] = Box(self.tokenizer.df[col].min(), self.tokenizer.df[col].max(), (1,))
            else:
                spaces[col] = Discrete(self.tokenizer.df[col].nunique())
        self.observation_space = Dict(spaces)

    def reset(self, **kwargs) -> np.ndarray | tuple[np.ndarray, dict]:
        """Plaecholder. Say something about the impact of self.test to starting point."""
        self.counter = 1
        nrows = range(self.tokenizer.df_shape[0])

        # Random starting index
        if self.test_mode:
            ix = 0  # Fixed index to 0 in test mode
        else:
            ix = np.random.randint(nrows.start, nrows.stop - self.reset_coldstart)
        self._ix = ix

        # Initialize trajectory
        self._simulation_array = self.tokenizer.df.iloc[
            self._ix : self._ix + self.reset_coldstart, :
        ].copy()
        self.simulation_token: np.ndarray = np.array([])
        # Extract state
        state = self._simulation_array.iloc[-1, self.tokenizer.state_indices].values

        if kwargs.get("return_info", False):
            return state, dict()
        else:
            return state

    @property
    def current_context(self):
        return self._simulation_array

    def _simulate_gpt_forward(self, action: list[str]) -> pd.DataFrame:
        """Perform a single logical steps and return new trajectory.

        Given the latest trajectory:
            (...s0,a0)

        Run a logical simulation step which is equal to gpt prediction (1 x array_size) steps
            (...s0,a0,r1)
            (...s0,a0,r1,s1)

        Append dummy a1',r2' of value [0,0] to maintain dataframe shape.
            (...s0,a0,r1,s1,a1',r2')

        Args:
            action list(str): Action

        Returns:
            pd.DataFrame: New trajectory after performing action
        """
        self._simulation_array.iloc[-1:, self.tokenizer.action_indices] = action

        simulation_sequence = self.tokenizer.field_tokenizer.transform(self._simulation_array)
        # Get context sequence up to action columns, as we are going to predict reward
        # and next states.
        seq = self.tokenizer.gpt_tokenize(
            simulation_sequence.sequence[: -self.tokenizer.reward_dim]
        )
        num_steps = self.tokenizer.reward_dim + self.tokenizer.state_dim

        next_step_gpt = self.gpt_sample_n_steps(
            seq,
            n_steps=num_steps,
            start_col_index=int(self.tokenizer.reward_indices[0]),
        )
        next_step = self.tokenizer.gpt_inverse_tokenize(next_step_gpt)

        # Get first token as dummy token for each action column
        action_cols = self.tokenizer.action_columns
        dummy_act_token = [
            self.tokenizer.field_tokenizer.valid_tokens_of_col_name(i)[0] for i in action_cols
        ]

        reward_cols = self.tokenizer.reward_columns
        dummy_reward_token = [
            self.tokenizer.field_tokenizer.valid_tokens_of_col_name(i)[0] for i in reward_cols
        ]

        new_sequence = np.append(next_step, np.array(dummy_act_token))
        new_sequence = np.append(new_sequence, np.array(dummy_reward_token))

        # Verify dataframe shape is retained
        if len(new_sequence) % self.tokenizer.column_len != 0:
            raise ValueError(
                "In one logical step, the number of prediction must match dataframe column size, "
                f"but last row len:{len(new_sequence)} "
                f"is not equal to column size:{self.tokenizer.column_len}"
            )

        self.simulation_token = np.array(new_sequence).reshape(-1, self.tokenizer.column_len)
        new_sequence_df = self.tokenizer.from_seq_to_dataframe(new_sequence)

        return new_sequence_df

    def _check_action(self, action: list[str]):
        """Raise exception when action string is not valid."""
        for idx, col in enumerate(self.tokenizer.action_columns):
            if action[idx] not in self.tokenizer.gym_action_to_enc[col].keys():
                raise ValueError(
                    f"action {action[idx]} ({type(action[idx])}) is not supported for column {col} "
                    f"({type(list(self.tokenizer.gym_action_to_enc[col].keys())[idx])})"
                    f'You can find the valid action from wi_df["{col}"].unique(). '
                    f""
                )

    def step(self, action: list) -> tuple[np.ndarray, float, bool, dict]:
        """Placeholder."""
        if not isinstance(action, list):
            raise TypeError(
                f"seq must be a list of action string, but got {type(action)}. E.g. ['a'], not [0]"
            )

        if len(action) != self.tokenizer.action_dim:
            raise TypeError(f"action dim is {self.tokenizer.action_dim}, but got {len(action)}")

        self._check_action(action)

        # Append trajectory with next step
        self._simulation_array = self._simulate_gpt_forward(action)

        # Retrieve latest next state/reward
        state = self._simulation_array.iloc[-1, self.tokenizer.state_indices].values
        reward = self._simulation_array.iloc[-2, self.tokenizer.reward_indices][0].astype(
            np.float32
        )
        self.counter += 1

        done = False
        if self.counter > self.max_steps:
            done = True

        return state, reward, done, {}

    def render(self, mode="human"):
        raise NotImplementedError("render() is not supported.")

    def _gpt_predict(self, seq: torch.Tensor, block_size: int) -> torch.Tensor:
        """Predict next GPT token given the input sequence of GPT tokens.

        Arguments:
            seq: GPT tokens, 2D dimension (n_sample, seq_length).
            block_size: maximum context window size.

        Returns:
            Logits for next GPT token. 2D dimension (n_sample, vocab_size).
        """
        if seq.dim() != 2:
            raise ValueError(f"seq must have dim of 2, but {seq.dim()} is given.")

        x_cond = seq if seq.size(1) <= block_size else seq[:, -block_size:]
        self.model.eval()
        logits = model_forward(self.model, x_cond)
        # extract transformer right most last token
        logits = logits[:, -1, :]
        return logits

    def _validate_logits(self, logits: torch.Tensor, cur_col_index: int, temperature: float = 1.0):
        """Filter out invalid GPT token index based on dataframe column.

        As example, when predicting the reward, this function will mask out the non-reward tokens.

        Arguments:
            logits: GPT model output logits.  2D dimension (n_sample, vocab_size).
            cur_col_index: Dataframe column index to be validated.

        Returns:
            Logits for next GPT token. 2D dimension (n_sample, vocab_size).
        """
        if not logits.dim() == 2:
            raise ValueError(f"Logits dim must be equal to 2, but got {logits.dim()}")

        # temperature scaling,
        logits = logits / temperature
        # Suppress invalid token idx
        eligible_indices = get_valid_gpt_token_idx(
            self.tokenizer._col_eligible_index, cur_col_index, self.tokenizer.simulator_ds
        )
        logits = logits_correction(logits, eligible_indices)

        return logits

    def _gpt_predict_filter(
        self, seq: torch.Tensor, cur_col_index: int, top_k: int | None = None
    ) -> torch.Tensor:
        """Predict next GPT token given the input GPT tokens, but filter out invalid token.

        Arguments:
            seq: Input context (list GPT token), 2D dimension (n_sample, seq_length).
            cur_col_index: Dataframe column to be predicted
            top_k: filter out top k if not None

        Returns:
            Logits for next token. 2D dimension (n_sample, vocab_size).
        """
        if seq.dim() != 2:
            raise ValueError(f"seq dim must be equal to 2, but got {seq.dim()}")

        logits = self._gpt_predict(seq, self.tokenizer.block_size)
        logits = self._validate_logits(logits, cur_col_index)

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        return logits

    def _gpt_logits_to_token(self, logits: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Convert logits to GPT token by doing a sampling.

        Arguments:
            logits: GPT model output logits.  2D dimension (n_sample, vocab_size).
            block_size: maximum context window size.

        Returns:
            Predicted next GPT token. 2D dimension (n_sample, 1s).
        """
        if not logits.dim() == 2:
            raise ValueError(f"Logits dim must be equal to 2, but got {logits.dim()}")

        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        return ix

    def gpt_sample(self, seq: np.ndarray, cur_col_index: int, sample: bool = True) -> np.ndarray:
        """Predict the next GPT token given the input GPT tokens.

        Arguments:
            seq: This is GPT token sequence. (Not dataframe tokenized sequence)
            cur_col_index: Inform which column index to be predicted, so that filtering can be done
                to remove invalid token.

        Returns:
            Predicted next GPT token.
        """

        if not isinstance(seq, np.ndarray):
            raise TypeError(f"seq must be a numpy array, but got {type(seq)}.")

        seq_ndim = seq.ndim
        if seq_ndim not in [1, 2]:
            raise ValueError(f"seq must have dim of 1 or 2, but {seq.ndim} is given.")

        max_gpt_token_index = max(self.tokenizer.tokenized_val_to_gpt_token_map.values())
        max_seq_token_index = seq.max().item()

        if not max_seq_token_index <= max_gpt_token_index:
            raise ValueError(
                f"seq has value {max_seq_token_index} that is greater than the known max GPT "
                f"token value {max_gpt_token_index}. Ensure input seq is GPT token, "
                "not dataframe token value."
            )
        if seq_ndim == 1:
            seq_tensor = torch.tensor(seq, dtype=torch.long)[None, ...]
        else:
            seq_tensor = torch.tensor(seq, dtype=torch.long)
        if self.manage_tensor_placement and self.device != "cpu":
            seq_tensor = seq_tensor.to(self.device)

        logits = self._gpt_predict_filter(seq_tensor, cur_col_index)
        token = self._gpt_logits_to_token(logits, sample)

        new_seq = token[0] if seq_ndim == 1 else token
        if self.manage_tensor_placement and self.device != "cpu":
            new_seq = new_seq.cpu()

        return new_seq.numpy()

    def gpt_sample_n_steps(
        self,
        seq: np.ndarray,
        n_steps: int,
        start_col_index: int,
        sample: bool = True,
    ) -> np.ndarray:
        """Given a GPT token sequence, sample the next ``n_steps`` of GPT tokens.

        Arguments:
            seq: This is GPT token sequence as a 1D array.
            n_steps: Number of next token to predict.
            start_col_index: Indicate the starting dataframe column index.

        Returns:
            Concatenated of original sequence with new predicted sequence.
        """

        if seq.ndim not in [1, 2]:
            raise ValueError(f"seq must have dim of 1 or 2, but {seq.ndim} is given.")

        col_length: int = self.tokenizer.column_len

        block_size = self.tokenizer.block_size
        if not block_size % col_length == 0:
            raise ValueError(
                f"block_size {block_size} % col_length {col_length} must be equal to 0, "
                f"but got {block_size % col_length}."
            )

        cur_col_index = start_col_index
        for k in range(n_steps):
            ix = self.gpt_sample(seq, cur_col_index, sample)
            seq = np.hstack([seq, ix])

            cur_col_index = (cur_col_index + 1) % col_length

        return seq

    def _handle_unseen_token(self, seq: np.ndarray) -> np.ndarray:
        """
        Map unseen dataframe token to nearest known dataframe token.

        As it is possible for user to pass in new context sequence that is unseen by GPT model,
        there is a need to map new value to nearest known dataframe token.

        """
        seq = seq.ravel()
        valid_token = list(self.tokenizer._gpt_token_to_tokenized_val_map.values())
        neighbors_idx = self.tokenizer.token_neighbors.kneighbors(
            seq.reshape(-1, 1), return_distance=False
        )
        return np.array([valid_token[i] for i in neighbors_idx.ravel()])

    def sample(
        self,
        seq: np.ndarray,
        max_size: int = 3,
        as_token: bool = False,
        correct_unseen_token: bool = True,
    ) -> wi.WiDataFrame:
        """
        Given a batch of context, perform one step sampling for actions and rewards.

        **Example:**

        Input: ::

            seq = [[1,2], [3,4]]
            max_size = 2

        Output: ::

            wi.WiDataFrame([]
                [10, 11], # From context [1,2]
                [12, 13], # From context [1,2]
                [20, 21], # From context [3,4]
                [22, 23], # From context [3,4]
            ])

        Args:
            seq: a batch of context (s, a, r, ..., s). Must end with states dataframe token.
                Shape is (batch_size, context_length).
            max_size: Number of samples to return.
            as_token: whether the returned dataframe should be in tokenized format, or in the
                original value space (approximated).
            correct_unseen_token: Map unseen token to the closest valid token when True.

        Returns:
            Whatif dataframe where each row is a sample with actions and rewards columns. The
            ``as_token`` determines whether the dataframe contents are tokenized or in the
            original value space (approximated).

            Shape is (batch_size * max_size, ...). Starting with 1st context's actions, followed
            2nd context's actions and so on.

        .. note::
            - Ensure the correct context sequence ``s, a, r, ..., s)`` is passed in.
            - Return ``max_size`` of sampling for each context. Result may not be unique.
            - Each rows of return result represent actions, rewards and values.


        """
        if not isinstance(seq, np.ndarray):
            raise TypeError(f"seq must be a numpy array, but got {type(seq)}")

        if seq.ndim == 1:
            # For backward compatibility
            seq = seq.reshape(1, -1)

        if seq.ndim != 2:
            raise ValueError(f"seq for batch_sample must have a dim of 2, but {seq.ndim} is given.")

        if seq.shape[1] > self.tokenizer.block_size:
            warnings.warn(
                f"Sequence is truncated as its length ({seq.shape[1]}) is greater "
                f"than block_size ({self.tokenizer.block_size})",
                UserWarning,
            )

        context_batch_size = len(seq)
        if correct_unseen_token:
            seq = self._handle_unseen_token(seq).reshape(context_batch_size, -1)

        start_col_index = self.tokenizer.action_indices[0]
        total_step = (
            self.tokenizer.action_dim + self.tokenizer.reward_dim + self.tokenizer.state_dim
        )
        seq_gpt = self.tokenizer.gpt_tokenize(seq.ravel()).reshape(context_batch_size, -1)
        seq_gpt = seq_gpt.repeat(max_size, axis=0)  # ([c1,c2], ...) -> ([c1,c1...,c2,c2...], ...)

        new_seq_gpt = self.gpt_sample_n_steps(
            seq=seq_gpt,
            n_steps=total_step,
            start_col_index=start_col_index,
            sample=True,
        )
        new_seq = self.tokenizer.gpt_inverse_tokenize(new_seq_gpt.ravel()).reshape(
            new_seq_gpt.shape[0], -1
        )

        action_stop = -total_step + self.tokenizer.action_dim
        reward_stop = action_stop + self.tokenizer.reward_dim
        actions = new_seq[:, -total_step:action_stop]
        rewards = new_seq[:, action_stop:reward_stop]
        next_states = new_seq[:, reward_stop:]
        samples = np.hstack([actions, rewards, next_states])

        df_ars = wi.WiDataFrame(
            samples,
            **self.tokenizer.df_tokenized.sar_d,
            columns=[
                *self.tokenizer.df_tokenized.actions,
                *self.tokenizer.df_tokenized.rewards,
                *self.tokenizer.df_tokenized.states,
            ],
        )
        df_sar = df_ars[df_ars.sar]
        if not as_token:
            df_sar = self.tokenizer.field_tokenizer.inverse_transform(df_sar)

        # Return actions and rewards only
        return df_sar.iloc[:, len(df_sar.states) :]

    def _check_lookahed_data(self, seq, action) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(seq, np.ndarray):
            raise TypeError(f"seq must be a numpy array, but got {type(seq)}")

        if seq.ndim == 1:
            # For backward compatibility
            seq = seq.reshape(1, -1)

        if seq.ndim != 2:
            raise ValueError(f"seq for batch_sample must have a dim of 2, but {seq.ndim} is given.")

        if isinstance(action, list):  # backward compatibility
            action = np.array(action)

        if not isinstance(action, np.ndarray):
            raise TypeError(f"action must be a numpy array, but got {type(seq)}")

        if action.ndim == 1:
            # For backward compatibility
            action = action.reshape([1, -1])

        if action.ndim != 2:
            raise TypeError(f"action must have dim of 2, but {action.ndim} is given")

        if action.shape[1] != self.tokenizer.action_dim:
            raise ValueError(
                f"The action dim should be {self.tokenizer.action_dim}, but got {action.shape[1]}"
            )
        return seq, action

    def lookahead(
        self,
        seq: np.ndarray,
        action: np.ndarray,
        correct_unseen_token: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a batch of context, and a batch of actions, simulates the expected rewards
        and next states for all combination of contexts and actions.

        This is a simulated step to get the estimated reward and next step,
        it can be run multiple times for planning purpose.

        **Examples 1 - Rewards and action have dim of 2**

        Input: ::

            seq = np.array([[1,2], [3,4]])
            action = np.array([[10,20], [30,40]])

        Output: ::

            reward = np.array([
                            [80, 81], # From seq = [1,2], action = [10,20]
                            [82, 83], # From seq = [1,2], action = [30,40]
                            [90, 91], # From seq = [3,4], action = [10,20]
                            [92, 93], # From seq = [3,4], action = [30,40]
                            ])

            next_states = np.array([
                                [180, 181], # From seq = [1,2], action = [10,20]
                                [182, 183], # From seq = [1,2], action = [30,40]
                                [190, 191], # From seq = [3,4], action = [10,20]
                                [192, 193], # From seq = [3,4], action = [30,40]
                                ])


        **Examples 2 - Reward has dim of 1, action is a list**

        Input: ::

            seq = np.array([1,2])
            action = [10,20]

        Output: ::

            reward = np.array([80, 81])
            next_states = np.array([180, 181])

        Args:
            seq: Context (s, a, r, ..., s). Must end with states dataframe token.
            action: Action dataframe token to be performed.
            correct_unseen_token: Map unseen token to the closest valid token when True.

        Returns:
            Return rewards array, and next states array.

        """
        seq_ndim_ori = seq.ndim
        action_ndim_ori = 1 if isinstance(action, list) else 2
        seq, action = self._check_lookahed_data(seq, action)

        context_batch_size = len(seq)
        action_batch_size = len(action)
        if correct_unseen_token:
            seq = self._handle_unseen_token(seq).reshape(context_batch_size, -1)
        seq_gpt = self.tokenizer.gpt_tokenize(seq.ravel()).reshape(context_batch_size, -1)
        seq_gpt = seq_gpt.repeat(action_batch_size, axis=0)
        action_gpt = self.tokenizer.gpt_tokenize(action.ravel()).reshape(action_batch_size, -1)
        action_gpt = np.tile(action_gpt, (context_batch_size, 1))
        seq_action_gpt = np.hstack([seq_gpt, action_gpt])

        # Predict reward and next states.
        num_steps = self.tokenizer.reward_dim + self.tokenizer.state_dim
        next_step_gpt = self.gpt_sample_n_steps(
            seq_action_gpt,
            n_steps=num_steps,
            start_col_index=int(self.tokenizer.reward_indices[0]),
        )
        new_seq = self.tokenizer.gpt_inverse_tokenize(next_step_gpt.ravel()).reshape(
            next_step_gpt.shape[0], -1
        )
        reward = new_seq[:, -num_steps : -self.tokenizer.state_dim]
        next_states = new_seq[:, -self.tokenizer.state_dim :]

        # For backward compatibility
        if seq_ndim_ori == 1 and action_ndim_ori == 1:
            reward = reward.flatten()
            next_states = next_states.flatten()

        return reward, next_states

    def get_valid_actions(self, seq: np.ndarray, max_size: int) -> wi.WiDataFrame:
        """Return a dataframe of sampled action tokens, given the input context.

        Arguments:
            seq: Input context sequence (1-dim) where context = (s, a, r, ..., s) which ends with
                state dataframe tokens.
            max_size: Number of samples to draw

        Returns:
            Whatif dataframe where each row denotes a sample with action columns, and the actions
            are in the tokenized forms.

        """
        if not isinstance(seq, np.ndarray):
            raise TypeError(f"seq must be a numpy array, but got {type(seq)}")

        result = self.sample(seq, max_size, as_token=True)
        return result[self.tokenizer.df_tokenized.actions]

    def _drop_invalid_dataframe_token(self, data_in: np.ndarray, cur_col_index: int) -> np.ndarray:
        """Remove invalid dataframe token from input array based on the column indicated."""

        valid_values = self.tokenizer.field_tokenizer.valid_tokens(int(cur_col_index))
        filter_values = [i for i in data_in if i in valid_values]
        return np.array(filter_values)

    def _convert_tensor_to_numpy_array(self, seq: torch.Tensor) -> np.ndarray:
        """Convert 2-dim tensor array into 1-dim numpy array."""

        if self.manage_tensor_placement and self.device != "cpu":
            temp = seq[0].cpu()
            new_seq = temp[0].numpy()
            return new_seq
        else:
            new_seq = seq[0].numpy()
            return new_seq

    def _convert_numpy_to_tensor_array(self, seq: np.ndarray) -> torch.Tensor:
        """Convert 1-dim numpy array into 2-dim tensor array on specific device."""
        new_seq = torch.tensor(seq, dtype=torch.long)[None, ...]
        if self.manage_tensor_placement and self.device != "cpu":
            new_seq = new_seq.to(self.device)
        return new_seq


def get_valid_gpt_token_idx(
    col_eligible_index: dict[int, list],
    current_col: int,
    sequence_dataset: SimulatorDataset,
) -> list[int]:
    """Return a list of valid GPT token index position.

    GPT model can return any token within the vocab size.
    This function is to identify the token that is valid for a given dataframe column.

    Arguments:
        current_col: Dataframe column index to check.
        col_eligible_index: Eligible dataframe token value range for each dataframe column

    Return:
        List of eligible GPT token index position.

    """
    max_col_idx = len(col_eligible_index)
    if not current_col < max_col_idx:
        raise ValueError(f"Column index {current_col=} out of range (max: {max_col_idx-1})")

    # Get the value range for selected dataframe column
    start, end = col_eligible_index[current_col]
    # Get the list of tokenized value
    idx_value = list(sequence_dataset.itos.values())
    # Get the index position for valid range
    eligible_indices = [idx for idx, val in enumerate(idx_value) if val >= start and val < end]
    return eligible_indices


def logits_correction(logits: torch.Tensor, eligible_indices: list[int]) -> torch.Tensor:
    """Update invalid token position logits to ``-np.inf`` based on predefined range.

    Arguments:
        eligible_indices: Eligible logits index position from ``get_valid_gpt_token_idx``

    Returns:
        torch.Tensor: New logits with invalid token position set to -np.inf
    """

    temp = torch.zeros_like(logits) + float("-Inf")
    temp[:, eligible_indices] = 0
    logits += temp
    return logits


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Update logits where all value set to ``-np.inf`` except for top k.

    Arguments:
        logits: Logits of 2D-dimension.
        k: Number of top result.

    Returns:
        New logits where all value set to -np.inf except for top k.
    """
    # This is karpathy/minGPT/mingpt/utils.py::top_k_logits(), but with added checks & docs.
    if logits.dim() != 2:
        raise ValueError(f"seq must have dim of 2, but {logits.dim()} is given.")

    if k > len(logits[0]):
        raise ValueError(
            f"topk {k} is greater than max of {len(logits[0])} of emb index, try to "
            "reduce to the maximum number of unique value for a column."
        )

    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


@dataclass
class SimulatorWrapper(gym.ActionWrapper):
    """Make :class:`a2rl.Simulator` accept tokenized actions only, to conform to the gym-style
    which requires discrete actions as integers.

    The gym-style requires the external agent to perform integer-encoded actions such as
    ``[0, 1, 2]``. On the other hand, ``Whatif`` :class:`Simulator` expects the original actions
    which may be non-integer (e.g., ``left``). To bridge these two styles, this wrapper
    automatically transforms the gym-style actions into Whatif-style actions, e.g., ``[0, 1, 2]``
    into ``['left' , 'right', 'up']``.

    Args:
        env: Whatif simulator which accept raw actions.
    """

    env: Simulator

    def __post_init__(self):
        if not isinstance(self.env, wi.Simulator):
            raise TypeError(f"The env must be Simulator type, but got {type(self.env)}")

        super().__init__(self.env)

    @staticmethod
    def _is_gym_actions(action: np.ndarray | list[int]) -> TypeGuard[np.ndarray | list[int]]:
        """Check whether ``action`` is either a 1D int array or a list of int."""
        if isinstance(action, list):
            return all(isinstance(x, int) for x in action)

        return is_npinstance(action, NDArray[Shape["*"], Integer])

    def action(self, action: np.ndarray | list[int]) -> list[str]:
        """Convert gym-style actions to their respective Whatif actions.

        Args:
            action: gym-style actions. Each action must be an integer in ``[0, actions_count)``.
        """
        if not self._is_gym_actions(action):
            raise TypeError(
                f"Expect actions as a list of int or array of np.integer, but got {action}."
            )

        new_action = []
        simulator = self.env
        tokenizer = simulator.tokenizer
        for i, a in enumerate(action):
            key = list(tokenizer.gym_enc_to_action.keys())[i]
            new_action.append(tokenizer.gym_enc_to_action[key][a])

        return new_action

    def reverse_action(self, action: list[str]):
        new_action = []
        simulator = self.env
        tokenizer = simulator.tokenizer
        for i, a in enumerate(action):
            key = list(tokenizer.gym_action_to_enc.keys())[i]
            new_action.append(tokenizer.gym_action_to_enc[key][a])

        return new_action

    @property
    def gym_action_to_enc(self):
        return self.env.tokenizer.gym_action_to_enc
