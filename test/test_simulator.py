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

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from pandas.api.types import is_numeric_dtype

import a2rl as wi
from a2rl.experimental.lightgpt import LightGPTBuilder, WarmupCosineLearningRateDecay
from a2rl.simulator import (
    TRAIN_CONFIG_KEYS,
    AutoTokenizer,
    Simulator,
    SimulatorDataset,
    SimulatorWrapper,
    get_valid_gpt_token_idx,
    logits_correction,
    top_k_logits,
)
from a2rl.utils import backtest


@pytest.fixture(scope="session")
def DUMMY_PATH(tmp_path_factory):
    return str(tmp_path_factory.mktemp("test_simulator"))


@pytest.fixture(scope="session")
def dummy_data() -> tuple[wi.WiDataFrame, wi.WiDataFrame, wi.WiDataFrame]:
    """
    Dummy df.trim():
       S1  S2 A1 A2    R
    0   5  10  a  x   20
    1   5  10  a  x   40
    2   5  50  b  y   50
    3  85  50  b  y   60
    4  85  90  c  z   80
    5  85  90  c  z  100
    """

    # Each row = [S2, S1, R, A2, A1]
    arr_input = [
        [10, 5, 20, "x", "a"],
        [10, 5, 40, "x", "a"],
        [50, 5, 50, "y", "b"],
        [50, 85, 60, "y", "b"],
        [90, 85, 80, "z", "c"],
        [90, 85, 100, "z", "c"],
    ]
    print(f"{arr_input=}")

    sar_d = dict(states=["S1", "S2"], actions=["A1", "A2"], rewards=["R"])
    df_input = wi.WiDataFrame(arr_input, columns=["S2", "S1", "R", "A2", "A1"], **sar_d)
    df_input.add_value()

    tok = wi.DiscreteTokenizer(num_bins_strategy="uniform").fit(df_input.trim())
    df_transformed = tok.transform(df_input.trim())
    df_reconstructed = tok.inverse_transform(df_transformed)

    print(f"{df_input=}")
    print(f"{df_transformed=}")
    print(f"{df_reconstructed=}")

    return df_input, df_transformed, df_reconstructed


@pytest.fixture
def df_dummy_reconstructed(dummy_data):
    return dummy_data[2]


@pytest.fixture
def df_dummy_tokenized(dummy_data):
    return dummy_data[1]


@pytest.fixture
def df_dummy(dummy_data):
    return dummy_data[0]


@pytest.fixture
def df_dummy_trimmed(dummy_data):
    return dummy_data[0].trim()


@pytest.fixture(scope="session")
def autotokenizer_dummy(dummy_data):
    field_tokenizer = wi.DiscreteTokenizer(num_bins_strategy="uniform")
    tokenizer = AutoTokenizer(dummy_data[0], block_size_row=2, field_tokenizer=field_tokenizer)
    return tokenizer


@pytest.fixture(scope="session", autouse=True)
def builder_mingpt(autotokenizer_dummy, DUMMY_PATH):
    torch.manual_seed(99)
    print("Creating dummy model...")
    with mock.patch("a2rl.simulator.BaseBuilder.config_loader") as mock_config:
        config = dict(
            epochs=1,
            batch_size=512,
            embedding_dim=512,
            gpt_n_layer=1,
            gpt_n_head=1,
            learning_rate=6e-4,
            num_workers=0,
            lr_decay=True,
        )
        mock_config.return_value = {"train_config": config}
        model_dir = Path(DUMMY_PATH) / "mingpt"
        builder = wi.GPTBuilder(autotokenizer_dummy, model_dir)
        builder.fit()
        for fname in (builder.model_name, "tokenizer.pt", "config.yaml"):
            assert (model_dir / fname).exists()
    return builder


@pytest.fixture(scope="session", autouse=True)
def builder_lightgpt(autotokenizer_dummy, DUMMY_PATH):
    import pytorch_lightning as pl

    torch.manual_seed(99)
    print("Creating dummy model...")
    with mock.patch("a2rl.simulator.BaseBuilder.config_loader") as mock_config:
        config = dict(
            epochs=1,
            batch_size=512,
            embedding_dim=512,
            gpt_n_layer=1,
            gpt_n_head=1,
            learning_rate=6e-4,
            num_workers=1,
            lr_decay=True,
        )
        mock_config.return_value = {"train_config": config}

        # Number of tokens backpropped in 1x iteration
        epoch_tokens = len(autotokenizer_dummy.train_dataset)

        lr_decay = WarmupCosineLearningRateDecay(
            learning_rate=6e-4,
            warmup_tokens=epoch_tokens // 2,
            final_tokens=config["epochs"] * epoch_tokens,
        )

        model_dir = Path(DUMMY_PATH) / "lightgpt"
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            benchmark=False,
            max_epochs=1,
            gradient_clip_val=1.0,
            callbacks=[lr_decay, pl.callbacks.ModelSummary(max_depth=2)],
            default_root_dir=model_dir,
        )
        builder = LightGPTBuilder(
            autotokenizer_dummy,
            kw_args={"trainer": trainer},
            model_dir=model_dir,
        )
        builder.fit()
        for fname in (builder.model_name, "tokenizer.pt", "config.yaml"):
            assert (model_dir / fname).exists()
    return builder


@pytest.fixture(scope="function")
def sim_mingpt(autotokenizer_dummy, builder_mingpt):
    env = Simulator(autotokenizer_dummy, builder_mingpt.model, max_steps=3, reset_coldstart=2)
    return env


@pytest.fixture(scope="function")
def sim_lightgpt(autotokenizer_dummy, builder_lightgpt):
    env = Simulator(autotokenizer_dummy, builder_lightgpt.model, max_steps=3, reset_coldstart=2)
    return env


#############################
# SimulatorDataset
#############################
@pytest.mark.parametrize(
    # fmt:off
    "block_size, expected_x, expected_y",
    [
        (
            3,  # context window
            np.array([0, 0, 1]),
            np.array([0, 1, 2]),
        ),
        (
            4,  # context window
            np.array([0, 0, 1, 2]),
            np.array([0, 1, 2, 3]),
        )
    ],
    # fmt:on
)
def test_simulator_dataset(block_size, expected_x, expected_y):
    input_seq = np.array([0, 0, 1, 2, 3, 4, 4, 5])
    expected_vocab_size = 6

    # Test wrong shape
    with pytest.raises(TypeError, match=r"Expect sequence as a 1D "):
        SimulatorDataset(input_seq.reshape(1, -1), block_size)

    # Test wrong dtype
    with pytest.raises(TypeError, match=r"Expect sequence as a 1D "):
        SimulatorDataset(input_seq.astype(float), block_size)

    ds = SimulatorDataset(input_seq, block_size)
    assert ds.block_size == block_size
    assert ds.vocab_size == expected_vocab_size

    idx = 0
    x, y = ds[idx]
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)
    assert (x.numpy() == expected_x).all()
    assert (y.numpy() == expected_y).all()


#############################
# AutoTokenizer
#############################
def test_autotokenizer_attrs(dummy_data):
    df = dummy_data[0]

    reversed_cols = list(reversed(df.columns))
    field_tokenizer = wi.DiscreteTokenizer(num_bins_strategy="uniform")
    tokenizer = AutoTokenizer(df[reversed_cols], block_size_row=2, field_tokenizer=field_tokenizer)
    assert tokenizer.gym_action_to_enc == {
        "A1": {"a": 0, "b": 1, "c": 2},
        "A2": {"x": 0, "y": 1, "z": 2},
    }
    assert tokenizer.gym_enc_to_action == {
        "A1": {0: "a", 1: "b", 2: "c"},
        "A2": {0: "x", 1: "y", 2: "z"},
    }

    print(tokenizer.col_eligible_index)
    assert tokenizer.col_eligible_index == {
        0: [0, 100],
        1: [100, 200],
        2: [400, 403],
        3: [403, 406],
        4: [200, 300],
        5: [300, 400],
    }
    assert tokenizer.state_columns == ["S1", "S2"]
    assert tokenizer.action_columns == ["A1", "A2"]
    assert tokenizer.reward_columns == ["R", "value"]
    assert tokenizer.column_len == df.shape[1]

    # Check tokenized df
    df_transformed_result = tokenizer.df_tokenized
    print(df_transformed_result)
    assert len(df_transformed_result.sequence) == np.multiply(*df.shape)
    assert df_transformed_result.shape == df.shape
    assert df_transformed_result.columns.tolist() == ["S1", "S2", "A1", "A2", "R", "value"]
    assert df_transformed_result.equals(dummy_data[1])
    # Check reconstructed df
    df_reconstructed_result = tokenizer.from_seq_to_dataframe(df_transformed_result.sequence)
    print(df_reconstructed_result)
    assert len(df_reconstructed_result.sequence) == np.multiply(*df.shape)
    assert df_reconstructed_result.shape == df.shape
    assert df_reconstructed_result.columns.tolist() == ["S1", "S2", "A1", "A2", "R", "value"]
    assert df_reconstructed_result.equals(dummy_data[2])

    df_reconstructed_result_inv = tokenizer.from_seq_to_dataframe(
        df_transformed_result.sequence, inverse=False
    )
    print(df_reconstructed_result_inv)
    assert len(df_reconstructed_result_inv.sequence) == np.multiply(*df.shape)
    assert df_reconstructed_result_inv.shape == df.shape
    assert df_reconstructed_result_inv.columns.tolist() == ["S1", "S2", "A1", "A2", "R", "value"]
    assert df_reconstructed_result_inv.equals(dummy_data[1])


def test_autotokenizer_blocksize_exception(df_dummy):
    print(df_dummy)
    with pytest.raises(ValueError):
        AutoTokenizer(df_dummy, block_size_row=10)


def test_autotokenizer_gpt_tokenize(autotokenizer_dummy):
    input_df_token = np.array([0, 99, 199, 300])
    expected_gpt_token = np.array([0, 1, 4, 11])
    tk = autotokenizer_dummy
    print(tk.tokenized_val_to_gpt_token_map)
    tokenized = tk.gpt_tokenize(input_df_token)
    assert isinstance(tokenized, np.ndarray)

    detokenized = tk.gpt_inverse_tokenize(tokenized)
    assert isinstance(detokenized, np.ndarray)

    assert np.all(tokenized == expected_gpt_token)
    assert np.all(detokenized == input_df_token)

    with pytest.raises(TypeError, match="seq must be a numpy array"):
        tk.gpt_tokenize(1)

    with pytest.raises(TypeError, match="seq must be a numpy array"):
        tk.gpt_inverse_tokenize(1)

    inv_input_seq = np.array([999, 999, 999, 999])
    with pytest.raises(ValueError, match="There is dataframe token .* that does not exist"):
        tk.gpt_tokenize(inv_input_seq)

    with pytest.raises(ValueError, match="There is GPT token .* that does not exist"):
        tk.gpt_inverse_tokenize(inv_input_seq)


def test_autotokenizer_from_seq_to_dataframe_exceptions(autotokenizer_dummy):
    tk = autotokenizer_dummy
    data = np.array([0, 150, 400, 403, 200, 300, 0])

    with pytest.raises(TypeError, match="seq must be a numpy array"):
        tk.from_seq_to_dataframe(1)

    with pytest.raises(ValueError, match="seq length must be multiple of column length"):
        tk.from_seq_to_dataframe(data)


#############################
# Simulator
#############################


def test_print_info(autotokenizer_dummy):
    tk = autotokenizer_dummy
    print()
    print(tk.df)
    print(tk.df_tokenized)
    print(tk.field_tokenizer.inverse_transform(tk.df_tokenized))
    print(tk.df_tokenized.sequence[:10])
    print(tk.col_eligible_index)
    print(tk.gpt_token_to_tokenized_val_map)
    print("Valid: dataframe token", tk.gpt_token_to_tokenized_val_map.values())
    print("Valid: GPT token", tk.gpt_token_to_tokenized_val_map.keys())


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_handle_unseen_token(sim):
    unseen_tokens = np.array([98, 140, 201])
    expected = np.array([99, 150, 200])
    result = sim._handle_unseen_token(unseen_tokens)
    assert np.all(np.equal(result, expected))


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_reset(sim):
    state_info = sim.reset(return_info=True)
    assert (
        isinstance(state_info, tuple)
        and len(state_info) == 2
        and isinstance(state_info[0], np.ndarray)
        and isinstance(state_info[1], dict)
    )

    state = sim.reset()
    assert isinstance(state, np.ndarray)
    assert len(state) == len(sim.tokenizer.state_indices)
    assert state.dtype == np.object
    assert sim._ix == 0


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_reset_context(sim, df_dummy_trimmed):
    # In test mode, the starting index is always 0, must map to 1st row of dataframe
    actual = sim.current_context.iloc[0]
    assert (actual == df_dummy_trimmed.iloc[0]).all()


def test_sim_reset_test_mode(autotokenizer_dummy):
    mock_model = mock.Mock()
    np.random.seed(99)
    env = Simulator(
        autotokenizer_dummy, mock_model, max_steps=3, reset_coldstart=2, test_mode=False
    )
    # Start with random index
    assert env._ix == 1


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_step_multidim(sim):
    env = sim
    seq = env.current_context
    new_seq = env._simulate_gpt_forward(["a", "x"])
    # Increase by one row in dataframe
    assert len(seq) + 1 == len(new_seq)
    # Maintain dataframe column size
    assert new_seq.shape[1] == env.tokenizer.column_len


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_counter(sim):
    assert sim.counter == 1
    sim.step(["a", "x"])
    # Counter increment for every step
    assert sim.counter == 2
    sim.reset()


@pytest.mark.parametrize(
    "max_steps, reset_coldstart",
    [
        (100, 2),
        (1, 5),
    ],
)
def test_sim_config(autotokenizer_dummy, max_steps, reset_coldstart):
    model = mock.Mock()
    env = Simulator(autotokenizer_dummy, model, max_steps, reset_coldstart)
    assert env.max_steps == max_steps
    assert env.reset_coldstart == reset_coldstart


@pytest.mark.parametrize(
    "current_col, expected",
    [
        (0, [0, 1]),
        (1, [2, 3]),
        (2, [4, 5]),
    ],
)
def test_sim_get_valid_gpt_token_idx(current_col, expected):
    col_eligible_index = {0: [0, 100], 1: [100, 200], 2: [200, 300]}
    test_ds = SimulatorDataset(np.array([10, 20, 120, 140, 201, 202]), block_size=2)

    # Check GPT model token index to dataframe tokenized value mapping
    assert test_ds.itos == {0: 10, 1: 20, 2: 120, 3: 140, 4: 201, 5: 202}
    # Check only valid GPT token idx is return
    result = get_valid_gpt_token_idx(col_eligible_index, current_col, test_ds)
    assert result == expected


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "cur_col_index, expected",
    [
        (0, [0, 1]),
        (1, [100, 101]),
        (2, [400]),
        (3, [403]),
        (4, [200, 201]),
        (5, [300, 301]),
    ],
)
def test_sim_drop_invalid_dataframe_token(sim, cur_col_index, expected):
    data = np.array([0, 1, 100, 101, 200, 201, 300, 301, 400, 403])
    result = sim._drop_invalid_dataframe_token(data, cur_col_index)
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "eligible_indices, logits, expected",
    [
        (
            [0, 1],
            torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            torch.tensor([[0.1, 0.2, -float("Inf"), -float("Inf"), -float("Inf")]]),
        ),
        (
            [2, 3, 4],
            torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            torch.tensor([[-float("Inf"), -float("Inf"), 0.3, 0.4, 0.5]]),
        ),
    ],
)
def test_logits_correction(logits, eligible_indices, expected):
    result = logits_correction(logits, eligible_indices)
    assert (torch.eq(result, expected)).all()


@pytest.mark.parametrize(
    "k, logits, expected",
    [
        (
            2,
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[-float("Inf"), 0.2000, 0.3000]]),
        ),
        (
            3,
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.1000, 0.2000, 0.3000]]),
        ),
    ],
)
def test_top_k_logits(k, logits, expected):
    new_logits = top_k_logits(logits, k)
    assert (torch.eq(new_logits, expected)).all()


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_convert_tensor_to_numpy_array(sim):
    seq_tensor = torch.tensor([[1, 2, 3]])
    seq = sim._convert_tensor_to_numpy_array(seq_tensor)
    assert seq.ndim == 1
    assert isinstance(seq, np.ndarray)
    assert np.all(seq == seq_tensor.numpy())


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_convert_numpy_to_tensor_array(sim):
    seq = np.array([1, 2, 3])
    seq_tensor = sim._convert_numpy_to_tensor_array(seq)
    if sim.manage_tensor_placement and sim.device != "cpu":
        seq_tensor = seq_tensor.cpu()
    assert seq_tensor.dim() == 2
    assert isinstance(seq_tensor, torch.Tensor)
    assert np.all(seq == seq_tensor.numpy())


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "seq, top_k",
    [
        (np.array([0, 100]), 2),
        (np.array([0, 100]), 3),
    ],
)
def test_sim_get_valid_actions_multidim(sim, seq, top_k, df_dummy_tokenized):
    torch.manual_seed(99)
    valid_actions = sim.get_valid_actions(seq, top_k)
    assert valid_actions.columns.tolist() == ["A1", "A2"]
    assert len(valid_actions) == top_k
    assert valid_actions["A1"].isin(df_dummy_tokenized["A1"]).all()
    assert valid_actions["A2"].isin(df_dummy_tokenized["A2"]).all()


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "tokenizer_data",
    [
        [[0, 2, 16, 18, 5, 11]],
        [[0, 2, 16, 18, 5, 11], [0, 2, 16, 18, 5, 11]],
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        3,
        10,
    ],
)
def test_sim_gpt_predict(sim, block_size, tokenizer_data):
    # Number of unique tokenized values
    vocab_size = sim.tokenizer.vocab_size

    # Take a short GPT token sequence
    tokenizer_data = torch.tensor(tokenizer_data)
    if sim.manage_tensor_placement:
        tokenizer_data = tokenizer_data.to(sim.device)
    sample_size = tokenizer_data.shape[0]
    # Do a next token prediction based on context windows
    result = sim._gpt_predict(tokenizer_data, block_size)
    print(f"{tokenizer_data=}")
    print(f"{vocab_size=}")
    print(f"{sample_size=}")
    # Result should match (n_sample, vocab_size)
    assert result.shape[0] == sample_size
    assert result.shape[1] == vocab_size


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "top_k",
    [
        1,
        3,
    ],
)
def test_sim_gpt_predict_filter(sim, top_k):
    # Tokenized value sequence [0, 100, 401, 403, 200, 300, 0, 100]
    # GPT token sequence [0, 2, 16, 18, 5, 11, 0, 2]

    cur_col_index = 2  # Predict 1st action
    gpt_token = [[0, 2, 16, 18, 5, 11, 0, 2]]
    gpt_token = torch.tensor(gpt_token)
    if sim.manage_tensor_placement:
        gpt_token = gpt_token.to(sim.device)
    result = sim._gpt_predict_filter(gpt_token, cur_col_index, top_k)
    print(result)
    valid_count = len(result[result != -np.inf])
    assert valid_count == top_k


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_gpt_logits_to_token(sim):
    logits = torch.tensor([[0.1, 0.2, 10], [10, 0.2, 0.3]])
    expected = torch.tensor([[2], [0]])
    token = sim._gpt_logits_to_token(logits, sample=True)
    result = torch.eq(token, expected)
    assert torch.all(result)


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "cur_col_index, expected_min, expected_max",
    [
        (0, 0, 1),
        (1, 2, 4),
        (2, 15, 17),
        (3, 18, 20),
        (4, 5, 10),
        (5, 11, 14),
    ],
)
def test_sim_gpt_sample(sim, cur_col_index, expected_min, expected_max):
    # Verify value filtering is done correctly for each column index
    torch.manual_seed(99)
    gpt_token = np.array([0, 2, 16, 18, 5, 11])
    preds = [sim.gpt_sample(gpt_token, cur_col_index, sample=True).item() for _ in range(50)]
    assert all([expected_min <= pred <= expected_max for pred in preds])


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "start_col_index, gpt_token_context",
    [
        (
            0,
            np.array([0, 2, 16, 18, 5, 11]),
        ),
        (
            2,
            np.array([0, 2, 16, 18, 5, 11, 0, 2]),
        ),
    ],
)
def test_sim_gpt_sample_n_steps(sim, start_col_index, gpt_token_context):
    NUM_STEP = 4
    result = sim.gpt_sample_n_steps(gpt_token_context, NUM_STEP, start_col_index)
    assert len(result) == NUM_STEP + len(gpt_token_context)
    for i in range(NUM_STEP):
        cur_idx = len(gpt_token_context) + i
        cur_value = result[cur_idx : cur_idx + 1]
        vald_token_range = get_valid_gpt_token_idx(
            sim.tokenizer.col_eligible_index,
            (start_col_index + i) % sim.tokenizer.column_len,
            sim.tokenizer.simulator_ds,
        )
        assert cur_value in vald_token_range


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "start_col_index, seed, gpt_token",
    [
        (0, 99, np.array([0, 1, 15, 18, 5, 11])),
        (0, 10, np.array([0, 1, 15, 18, 5, 11])),
    ],
)
def test_sim_gpt_sample_n_steps_sample_true(
    sim,
    start_col_index,
    seed,
    gpt_token,
    df_dummy_tokenized,
):
    results = sim.gpt_sample_n_steps(gpt_token, 4, start_col_index, sample=True)
    historical_context = results[:-4]
    pred = results[-4:]

    assert np.all(historical_context == gpt_token)

    # Always remember that gpt model returns stochastic predictions. Hence, the test must not check
    # for point equality, otherwise the test is very fragile to changes and we end-up chasing the
    # tail. Instead, acceptance criteria simply makes sure that gpt tokens are inverted to dataframe
    # tokens in valid range.
    pred2 = sim.tokenizer.gpt_inverse_tokenize(pred)
    min_valid_range = df_dummy_tokenized.iloc[:, :4].min().values
    max_valid_range = df_dummy_tokenized.iloc[:, :4].max().values
    assert np.all((pred2 >= min_valid_range) & (pred2 <= max_valid_range))


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "custom_context",
    [
        np.array([99, 199]),
        np.array([[99, 199]]),
        np.array([[99, 199], [0, 100]]),
    ],
)
@pytest.mark.parametrize(
    "action_seq",
    [
        np.array([402, 405]),
        np.array([[402, 405]]),
        np.array([[402, 405], [403, 405]]),
    ],
)
def test_sim_lookahead(sim, custom_context, action_seq, df_dummy_tokenized):
    torch.manual_seed(99)
    reward, next_states = sim.lookahead(custom_context, action_seq)
    assert isinstance(reward, np.ndarray)
    assert isinstance(next_states, np.ndarray)
    context_batch_size = len(custom_context) if custom_context.ndim != 1 else 1
    action_batch_size = len(action_seq) if action_seq.ndim != 1 else 1
    assert len(reward) == context_batch_size * action_batch_size
    assert len(next_states) == context_batch_size * action_batch_size
    assert np.all(np.isin(reward[:, 0], df_dummy_tokenized["R"]))
    assert np.all(np.isin(reward[:, 1], df_dummy_tokenized["value"]))
    assert np.all(np.isin(next_states[:, 0], df_dummy_tokenized["S1"]))
    assert np.all(np.isin(next_states[:, 1], df_dummy_tokenized["S2"]))


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "custom_context",
    [np.array([99, 199])],
)
@pytest.mark.parametrize(
    "action_seq",
    [[402, 405]],
)
def test_sim_lookahead_backware_compatible(sim, custom_context, action_seq, df_dummy_tokenized):
    torch.manual_seed(99)
    reward, next_states = sim.lookahead(custom_context, action_seq)
    assert isinstance(reward, np.ndarray)
    assert isinstance(next_states, np.ndarray)
    assert reward.ndim == 1
    assert next_states.ndim == 1
    assert np.all(np.isin(reward[0], df_dummy_tokenized["R"]))
    assert np.all(np.isin(reward[1], df_dummy_tokenized["value"]))
    assert np.all(np.isin(next_states[0], df_dummy_tokenized["S1"]))
    assert np.all(np.isin(next_states[1], df_dummy_tokenized["S2"]))


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize("max_size", [4])
@pytest.mark.parametrize(
    "custom_context",
    [
        np.array([99, 199, 402, 405, 275, 399, 0, 100]),
        np.array([[99, 199, 402, 405, 275, 399, 0, 100]]),
        np.array(
            [
                [0, 150, 400, 405, 275, 399, 0, 100],
                [99, 199, 402, 405, 275, 399, 0, 100],
            ]
        ),
    ],
)
def test_sim_sample(sim, max_size, custom_context, df_dummy_reconstructed):
    torch.manual_seed(99)
    result = sim.sample(custom_context, max_size=max_size, as_token=False)
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ["A1", "A2", "R", "value"]
    batch_size = len(custom_context) if custom_context.ndim != 1 else 1
    assert len(result) == max_size * batch_size
    assert (result["R"].isin(df_dummy_reconstructed["R"])).all()
    assert (result["value"].isin(df_dummy_reconstructed["value"])).all()
    assert result["A1"].isin(df_dummy_reconstructed["A1"]).all()
    assert result["A2"].isin(df_dummy_reconstructed["A2"]).all()


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize("max_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_sim_sample_as_token(sim, max_size, batch_size, df_dummy_tokenized):
    torch.manual_seed(99)
    custom_context = np.array([99, 199, 402, 405, 275, 399, 0, 100])
    custom_context = np.tile(custom_context, (batch_size, 1))
    result = sim.sample(custom_context, max_size, as_token=True)
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ["A1", "A2", "R", "value"]
    assert len(result) == max_size * batch_size
    assert result["A1"].isin(df_dummy_tokenized["A1"]).all()
    assert result["A2"].isin(df_dummy_tokenized["A2"]).all()


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_sample_unseen_context(sim):
    torch.manual_seed(99)
    unseen_context = np.array([2, 101])
    with pytest.raises(ValueError):
        sim.sample(unseen_context, 1, as_token=True, correct_unseen_token=False)


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_gym_env_attr(sim):
    sim = sim

    action_space = sim.action_space
    obs_space = sim.observation_space
    assert isinstance(action_space, MultiDiscrete)
    assert isinstance(obs_space, Dict)

    print(sim.tokenizer.gym_enc_to_action)
    expected_num_action = len(sim.tokenizer.gym_enc_to_action)
    assert len(action_space) == expected_num_action


@pytest.mark.parametrize(
    "sim",
    [
        pytest.lazy_fixture("sim_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("sim_lightgpt"),  # type: ignore[operator]
    ],
)
def test_backtest(df_dummy, sim):
    CONTEXT_ROWS = 2
    PREDICT_ROWS = 3
    result = backtest(df_dummy, sim, 0, CONTEXT_ROWS, PREDICT_ROWS)
    assert len(result) == (CONTEXT_ROWS + PREDICT_ROWS)
    assert np.all(result.columns == ["S1", "S2", "A1", "A2", "R", "value"])


@pytest.fixture(scope="session")
def dummy_data_str_states() -> wi.WiDataFrame:
    arr_input = [
        ["j", 5, 20, "x", "a"],
        ["j", 5, 40, "x", "a"],
        ["k", 5, 50, "y", "b"],
        ["k", 85, 60, "y", "b"],
        ["l", 85, 80, "z", "c"],
        ["l", 85, 100, "z", "c"],
    ]
    sar_d = dict(states=["S1", "S2"], actions=["A1", "A2"], rewards=["R"])
    df_input = wi.WiDataFrame(arr_input, columns=["S1", "S2", "R", "A2", "A1"], **sar_d)
    df_input.add_value()
    return df_input


def test_str_states(dummy_data_str_states, DUMMY_PATH):
    tokenizer = AutoTokenizer(dummy_data_str_states, 2)
    config = dict(
        epochs=1,
        batch_size=512,
        embedding_dim=512,
        gpt_n_layer=1,
        gpt_n_head=1,
        learning_rate=6e-4,
        num_workers=0,
        lr_decay=True,
    )
    train_config = {"train_config": config}
    model_dir = Path(DUMMY_PATH) / "mingpt"
    builder = wi.GPTBuilder(tokenizer, model_dir, train_config)
    model = builder.fit()
    sim = wi.Simulator(tokenizer, model)
    reset_result = sim.reset()
    assert isinstance(reset_result, np.ndarray)
    assert np.all(reset_result == tokenizer.df.iloc[0, tokenizer.state_indices].values)

    obs_result = sim.observation_space
    for i, col in enumerate(tokenizer.df.states):
        if is_numeric_dtype(tokenizer.df[col]):
            assert isinstance(obs_result[col], Box)
        else:
            assert isinstance(obs_result[col], Discrete)


#############################
# Simulator Wrapper
#############################


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_sim_wrapper_exception(df_dummy, builder):
    widf = df_dummy
    tokenizer = AutoTokenizer(widf, 2)
    simulator = Simulator(tokenizer, builder.model)

    with pytest.raises(TypeError, match="The env must be Simulator type"):
        not_sim_type = 1
        sim_wrapper = SimulatorWrapper(not_sim_type)

    sim_wrapper = SimulatorWrapper(simulator)
    with pytest.raises(TypeError, match="Expect actions as a list of int or array of np.integer"):
        not_list = 0
        sim_wrapper.action(not_list)

    with pytest.raises(TypeError):
        sim_wrapper.action(["x"], match="Expect actions as a list of int or array of np.integer")


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "action, expected",
    [
        ([0], ["a"]),
        ([0, 2], ["a", "z"]),  # This is 2-dim action (A1 and A2)
    ],
)
def test_sim_wrapper_action(df_dummy, builder, action, expected):
    widf = df_dummy
    tokenizer = AutoTokenizer(widf, 2)
    simulator = Simulator(tokenizer, builder.model)
    sim_wrapper = SimulatorWrapper(simulator)
    print(sim_wrapper.gym_action_to_enc)
    result = sim_wrapper.action(action)
    assert result == expected

    action_array = np.array(action)
    result = sim_wrapper.action(action_array)
    assert result == expected


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
@pytest.mark.parametrize(
    "action, expected",
    [
        (["a"], [0]),
        (["a", "z"], [0, 2]),  # This is 2-dim action
    ],
)
def test_sim_wrapper_reverse_action(df_dummy, builder, action, expected):
    widf = df_dummy
    tokenizer = AutoTokenizer(widf, 2)
    simulator = Simulator(tokenizer, builder.model)
    sim_wrapper = SimulatorWrapper(simulator)
    result = sim_wrapper.reverse_action(action)
    assert result == expected


#############################
# GPTBuilder
#############################


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_builder_load_model(builder):
    mock_tokenizer = mock.Mock()
    kw_args = {"trainer": mock.Mock()}
    builder = builder.__class__(mock_tokenizer, builder.model_dir, kw_args=kw_args)
    gpt_model = builder.load_model()
    assert isinstance(gpt_model, builder.model.__class__)


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_builder_load_model_exception(builder):
    with pytest.raises(FileNotFoundError):
        mock_tokenizer = mock.Mock()
        kw_args = {"trainer": mock.Mock()}
        builder = builder.__class__(mock_tokenizer, "invalid_path", kw_args=kw_args)
        builder.load_model()


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_builder_property(builder):
    mock_tokenizer = mock.Mock()
    kw_args = {"trainer": mock.Mock()}
    builder = builder.__class__(mock_tokenizer, builder.model_dir, kw_args=kw_args)
    with pytest.raises(ValueError, match="Please load the model"):
        builder.model


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_builder_config_loader(builder):
    # Load default config
    config = builder.config_loader()
    print(config)
    assert "train_config" in config

    train_config = config["train_config"]

    assert all(i in train_config for i in TRAIN_CONFIG_KEYS)
    # Config should have value
    for i in TRAIN_CONFIG_KEYS:
        assert train_config[i] is not None


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_builder_config_loader_param_config(builder, DUMMY_PATH):
    mock_tokenizer = mock.Mock()
    expected_config = {
        "train_config": {
            "epochs": 1,
            "batch_size": 512,
            "embedding_dim": 512,
            "gpt_n_layer": 1,
            "gpt_n_head": 1,
            "learning_rate": "6e-4",
            "num_workers": 0,
            "lr_decay": True,
        }
    }

    builder = builder.__class__(
        mock_tokenizer,
        DUMMY_PATH,
        expected_config,
        kw_args={"trainer": mock.Mock()},
    )
    config = builder.config_loader()
    assert config == expected_config


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_builder_config_loader_exception(builder):
    mock_tokenizer = mock.Mock()
    kw_args = {"trainer": mock.Mock()}
    with pytest.raises(FileNotFoundError):
        builder.__class__(mock_tokenizer, "dummy", "./src/a2rl/missing.yaml", kw_args=kw_args)

    with pytest.raises(ValueError):
        invalid_config = {"train_config": {"dummy": 40}}
        builder.__class__(mock_tokenizer, "dummy", invalid_config, kw_args=kw_args)


@pytest.mark.skip_slow
@pytest.mark.parametrize(
    "builder, num_workers",
    [
        (pytest.lazy_fixture("builder_mingpt"), 0),  # type: ignore[operator]
        (pytest.lazy_fixture("builder_lightgpt"), 1),  # type: ignore[operator]
    ],
)
def test_builder_fit(tmp_path, autotokenizer_dummy, builder, num_workers):
    kw_args = {"train_config": None}
    with mock.patch("a2rl.simulator.BaseBuilder.config_loader") as mock_config:
        config = dict(
            epochs=1,
            batch_size=512,
            embedding_dim=512,
            gpt_n_layer=1,
            gpt_n_head=1,
            learning_rate=6e-4,
            num_workers=num_workers,
            lr_decay=True,
        )
        mock_config.return_value = {"train_config": config}
        builder2 = builder.__class__(autotokenizer_dummy, tmp_path, config, kw_args=kw_args)
        # Overwrite to run 1 epoch for test
        model = builder2.fit()
        assert (tmp_path / builder2.model_name).exists()
        assert isinstance(model, builder.model.__class__)


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_builder_sample(autotokenizer_dummy, builder):
    tk = autotokenizer_dummy
    context_seq = tk.df_tokenized.sequence[:4]
    context_seq_gpt = tk.gpt_tokenize(context_seq)

    n_steps = 4
    print(f"{context_seq_gpt=} with {n_steps=}")
    new_seq_gpt = builder.sample(
        context_seq_gpt,
        n_steps=n_steps,
        temperature=1,
        sample=True,
        top_k=1,
    )
    print(f"{new_seq_gpt=}")
    # Check the type and shape
    assert isinstance(new_seq_gpt, np.ndarray)
    assert len(new_seq_gpt) == len(context_seq_gpt) + n_steps
    assert np.all(new_seq_gpt[: len(context_seq_gpt)] == context_seq_gpt)

    # Returned GPT tokens must be within the valid range.
    assert set(new_seq_gpt) - set(builder.tokenizer.gpt_token_to_tokenized_val_map) == set()


@pytest.mark.parametrize(
    "builder",
    [
        pytest.lazy_fixture("builder_mingpt"),  # type: ignore[operator]
        pytest.lazy_fixture("builder_lightgpt"),  # type: ignore[operator]
    ],
)
def test_builder_sample_exception(autotokenizer_dummy, builder):
    with pytest.raises(ValueError, match="Please make sure fit"):
        builder2 = builder.__class__(
            autotokenizer_dummy,
            builder.model_dir.parent / "test_builder_sample_exception" / builder.__class__.__name__,
            kw_args={"trainer": None},
        )
        builder2.sample(seq=1, n_steps=4, temperature=1, sample=False, top_k=False)

    with pytest.raises(TypeError, match="seq must be a numpy"):
        builder.sample(seq=1, n_steps=4, temperature=1, sample=False, top_k=False)

    with pytest.raises(ValueError, match="seq shape must have dim of 1"):
        builder.sample(seq=np.array([[1]]), n_steps=4, temperature=1, sample=False, top_k=False)

    with pytest.raises(ValueError, match="The model has not seen the seq dataset"):
        builder.sample(seq=np.array([99999]), n_steps=4, temperature=1, sample=False, top_k=False)
