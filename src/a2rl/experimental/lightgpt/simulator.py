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

from dataclasses import dataclass
from time import perf_counter

import pytorch_lightning as pl
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from a2rl.simulator import BaseBuilder

from .lr_decay import WarmupCosineLearningRateDecay
from .model import LightGPT


@dataclass
class LightGPTBuilder(BaseBuilder[LightGPT, pl.Trainer]):
    """High-level APIs to train and evaluate a Lightning-based GPT model based on the data loaded in
    :class:`~a2rl.AutoTokenizer`.

    It has no knowledge of dataframe shape, and which values belong to action/states/reward.

    Arguments:
        model_dir: Model directory.
        tokenizer: This is a :class:`~a2rl.AutoTokenizer`.
        config: Custom configuration file or dictionary. When set to ``None``, use the built-in
            configuration in ``a2rl/experimental/lightgpt/config.yaml``.

    Configuration file must meet the following ``yaml`` format.

    .. code-block:: yaml

        train_config:
            epochs: 5
            batch_size: 512
            embedding_dim: 512
            gpt_n_layer: 1
            gpt_n_head: 1
            learning_rate: 6e-4
            num_workers: 1
            lr_decay: True

    Examples
    --------
    Train a model, and save to a temporary directory.

        >>> import pytorch_lightning as pl
        >>> import a2rl as wi
        >>> from a2rl import AutoTokenizer
        >>> from a2rl.experimental.lightgpt import LightGPTBuilder, WarmupCosineLearningRateDecay

        >>> wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller"))
        >>> tokenizer = AutoTokenizer(wi_df, block_size_row=2)
        >>> with tempfile.TemporaryDirectory() as model_dir:
        ...     builder = LightGPTBuilder(tokenizer, model_dir, kw_args={"trainer": trainer})
        ...     model = builder.fit()  # doctest:+SKIP

    Train a model with a custom :class:`pytorch_lightning.Trainer`.

    .. code-block:: python

        >>> import pytorch_lightning as pl
        >>> import a2rl as wi
        >>> from a2rl import AutoTokenizer
        >>> from a2rl.experimental.lightgpt import LightGPTBuilder, WarmupCosineLearningRateDecay

        >>> wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller"))
        >>> tokenizer = AutoTokenizer(wi_df, block_size_row=2)
        >>> with tempfile.TemporaryDirectory() as model_dir:
        ...     # PyTorch Lightning stuffs. See Pytorch Lightning docs for more details.
        ...     max_epochs = 5  # Will ignore config.yaml
        ...     epoch_tokens = len(tokenizer.train_dataset)
        ...     lr_decay = WarmupCosineLearningRateDecay(
        ...         learning_rate=6e-4,
        ...         warmup_tokens=epoch_tokens // 2,
        ...         final_tokens=max_epochs * epoch_tokens,
        ...     )
        ...     trainer = pl.Trainer(
        ...         accelerator="auto",
        ...         devices="auto",
        ...         benchmark=False,
        ...         max_epochs=max_epochs,
        ...         gradient_clip_val=1.0,
        ...         callbacks=[lr_decay, pl.callbacks.ModelSummary(max_depth=2)],
        ...         default_root_dir=model_dir,
        ...     )
        ...
        ...     # Bread-and-butter stuffs (i.e., business-as-usual) with A2RL model builder.
        ...     builder = LightGPTBuilder(tokenizer, model_dir, kw_args={"trainer": trainer})
        ...     model = builder.fit()  # doctest:+SKIP

    The rest examples follow a similar structure to :class:`a2rl.GPTBuilder`, but remember to
    create and pass a PyTorch Lightning's trainer accordingly.
    """

    def __post_init__(self):
        # Announce to parent that Pytorch Lightning takes over this aspect.
        #
        # See: BaseBuilder.manage_tensor_placement
        self.manage_tensor_placement = False
        super().__post_init__()

        trainer = self.kw_args.get("trainer", None)
        if not trainer:
            epoch_tokens = len(self.tokenizer.train_dataset)
            lr_decay = WarmupCosineLearningRateDecay(
                learning_rate=6e-4,
                warmup_tokens=epoch_tokens // 2,
                final_tokens=self._configs["epochs"] * epoch_tokens,
            )
            callbacks = [lr_decay, pl.callbacks.ModelSummary(max_depth=2)]
            self.trainer = pl.Trainer(
                accelerator="auto",
                devices="auto",
                benchmark=False,
                max_epochs=self._configs["epochs"],
                gradient_clip_val=1.0,
                callbacks=callbacks if self._configs["lr_decay"] else callbacks[1:],
                default_root_dir=self.model_dir,
            )
        else:
            self.trainer = trainer

    def fit(self, validate: bool = True) -> LightGPT:
        """Start training model."""
        logger.info(self._configs)

        # Only expose commonly use configuration in config file.
        self._model = LightGPT(
            self.tokenizer.vocab_size,
            self.tokenizer.block_size,
            n_layer=self._configs["gpt_n_layer"],
            n_head=self._configs["gpt_n_head"],
            n_embd=self._configs["embedding_dim"],
        )

        # Set persistent_workers=True, otherwise noticable lags at the beginning of every
        # epoch (when num_workers > 0 OR with ddp* strategy).
        #
        # https://github.com/Lightning-AI/lightning/issues/10389#issuecomment-1077672897
        t1_start = perf_counter()
        train_dataloader = DataLoader(
            self.tokenizer.train_dataset,
            shuffle=True,
            batch_size=self._configs["batch_size"],
            num_workers=self._configs["num_workers"],
            persistent_workers=True,
        )
        if validate and len(self.tokenizer.test_dataset):
            test_dataloader = DataLoader(
                self.tokenizer.test_dataset,
                shuffle=False,
                batch_size=self._configs["batch_size"],
                num_workers=self._configs["num_workers"],
                persistent_workers=True,
            )
        else:
            test_dataloader = None
        self.trainer.fit(self._model, train_dataloader, test_dataloader)
        t1_stop = perf_counter()

        logger.info(f"Training time in mins: {(t1_stop - t1_start)/60:.02}")
        self.save_model()
        self._fitted = True
        return self.model
