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

import random
from datetime import datetime
from random import randrange
from typing import Any

import cloudpickle
import gym
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from matplotlib.pyplot import Axes
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm

import a2rl as wi


class NotMDPDataError(Exception):
    """Exception thrown when data does not exhibit MDP properties."""

    pass  # noqa


def timestamp() -> str:
    utc_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    salt = randrange(0x7FFFFFFF)  # nosec B311 -- this is not for cryptographic purpose.
    return f"{utc_ts}utc-{salt}"


def force_assert(condition: bool, msg: None | str = None):
    """Raise :exc:`AssertionError` when ``condition`` is ``False``.

    Use this judiciously (and conciously) when you want to always enforce assertion checks, even
    when Python runs in the :py:option:`optimized mode <-O>` (which ignores the
    :ref:`assert <python:assert>`
    """
    if not condition:
        raise AssertionError(msg)


def set_seed(seed):
    """Minimalistic implementation to fix random seeds in python.random, numpy, and pytorch.

    For a more robust implementation, you may want to consider
    :func:`pytorch_lightning.utilities.seed.seed_everything()`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pickle_save(path: str, obj: Any):
    """Serialized object."""
    with open(path, "wb") as handle:
        cloudpickle.dump(obj, handle)


def pickle_load(path: str) -> Any:
    """Deerialized object."""
    with open(path, "rb") as handle:
        obj = cloudpickle.load(handle)
    return obj


def backtest(
    df: wi.WiDataFrame,  # Full groundtruth data, non tokeinzed version
    simulator: wi.Simulator,
    start_row: int = 0,  # Context from groundtruth
    context_rows: int = 2,
    predict_rows: int = 3,
    return_groudtruth: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Utility to perform backtesting on simulator.

    .. note::
        By using groundtruth dataset ``df``, take ``context_rows`` number of rows as context,
        and groundtruth action, predict the next states ``(s1,s2...)`` and rewards ``(r,v)``.
        Then append the predicted next states and rewards as new context, and repeat the steps again
        until we get ``predict_rows`` number of new rows.

    Arguments:
        df: WiDataFrame, this is the original dataframe before tokenized.
        simulator: Pretrain whatif Simulator.
        start_row: The dataframe starting row index for backtest
        context_rows: Number of dataframe rows to be used as context, starting from ``start_row``.
        predict_rows: Number of dataframe rows to predict.
        return_groudtruth: Return groundtruth dataframe for comparison when true.

    Returns:
        Simulated Dataframe.
    """
    if len(df) < (start_row + context_rows + predict_rows):
        raise ValueError(
            f"The sum of start_row, context_rows and predict_rows, i.e. "
            f"{start_row + context_rows + predict_rows}, "
            f"should not be greater than num of rows in df {len(df) }."
        )
    # Retrieve test data from tokenizer used for training the simulator.
    df_test = simulator.tokenizer.field_tokenizer.transform(df)
    df_context = df_test.iloc[start_row : start_row + context_rows]

    curr_row_idx = start_row + context_rows - 1
    action_idx = simulator.tokenizer.action_indices

    context_end_pos = simulator.tokenizer.action_dim + simulator.tokenizer.reward_dim
    context = df_context.sequence[:-context_end_pos].reshape(1, -1)
    logger.info(f"Initial {context.shape=}")

    # INFO: debug is enabled for now to see backtest progress
    # Intentionally increment predict_rows as the last row of context will need to be
    # predicted as well to form full columns. It is more natural to have total number
    # of rows = context_rows + predict_rows.
    for i in range(predict_rows + 1):
        logger.info(f"Predicting row:{i+1}, {curr_row_idx=}")
        hist_action = df_test.iloc[curr_row_idx, action_idx].values.reshape(1, -1)
        logger.debug(f"{hist_action=}")
        reward, next_states = simulator.lookahead(context, hist_action)
        logger.debug(f"{reward=}, {next_states=}")

        new_context = np.concatenate([context, hist_action, reward, next_states], axis=1)
        logger.debug(f"{new_context.shape=}")
        context = new_context
        curr_row_idx += 1

    # Throw away last states
    new_sequence = context[0, : -simulator.tokenizer.state_dim]
    logger.debug(f"{new_sequence.shape=}")
    pred_df = simulator.tokenizer.from_seq_to_dataframe(new_sequence, True)
    gt_df = df.iloc[start_row : start_row + context_rows + predict_rows].reset_index(drop=True)

    if return_groudtruth:
        return pred_df, gt_df
    else:
        return pred_df


def assert_mdp(data: wi.WiDataFrame, lags: int = 10) -> None:
    """Assert that dataframe ``data`` has the MDP properties, and raise a :class:`NotMDPDataError`
    otherwise.

    Args:
        data: dataframe to check.
        lags: the number of distinct lags (0 to ``lags-1``) to test.
    """
    res: pd.DataFrame = markovian_matrix(data, lags=lags)
    bad_tests: pd.Series = res.idxmax(axis=1).isna()

    if not bad_tests.any():
        return

    bad_test_names = bad_tests.index[bad_tests].tolist()
    raise NotMDPDataError(f"These tests fail MDP checks: {bad_test_names}")


def plot_information(data, lags: int = 10) -> Axes:
    """Plot the results of MDP checks on dataframe ``data``.

    Args:
        data: dataframe to check and plot.

    Returns:
        The plot as a matplotlib axes.
    """
    res = markovian_matrix(data, lags=lags)
    print(res.idxmax(axis=1))
    return sns.heatmap(res, cmap="RdYlGn", linewidths=0.5, vmin=0, vmax=1)


def entropy(Y: np.ndarray) -> float:
    """The `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_ of the input
    series.

    Args:
        Y: tokenized input 1D array.

    Returns:
        The entropy
    """

    unique, count = np.unique(Y.astype("<U22"), return_counts=True, axis=0)
    prob = count / len(Y)

    en = np.sum((-1) * prob * np.log2(prob))

    return en


def tokenize(df: wi.WiDataFrame) -> np.ndarray:
    """Concats all columns to one.

    Args:
        df: a discretized dataframe.

    Returns:
        A sequence whose length equals to the number of rows in the input dataframe. Each element
        in the sequence is the concatenation of tokens of an input row.

    Examples
    --------

    .. code-block:: python

        >>> import a2rl as wi
        >>> from a2rl.utils import tokenize
        >>>
        >>> wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller")).trim()
        >>> wi_df = wi.DiscreteTokenizer().fit_transform(wi_df)
        >>> seq = tokenize(wi_df)

        >>> wi_df.shape
        (9153, 4)

        >>> seq.shape
        (9153,)

        >>> seq[:5]  # doctest: +SKIP
        array([61165305280, 44161305280, 59177305281, 32172305280, 59170305280])
    """
    return (
        pd.Series(df.fillna(df.median()).astype(int).astype(str).values.tolist())
        .str.join("")
        .replace(r"\D+", "")
        .values
    )


def conditional_entropy(Y: np.ndarray, X: np.ndarray, laplace_smoothing: bool = True) -> float:
    """The `conditional entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_ of
    the input series given a conditioning series H(Y|X).

    Args:
        Y: tokenized input 1D array. The entropy is calculated on this series.
        X: tokenized input 1D array. The conditioning array.

    Returns:
        The conditonal entropy
    """

    z = np.vstack((X, Y)).T
    z = z[z[:, 0].argsort()]
    groups = np.split(z[:, 1], np.unique(z[:, 0], return_index=True)[1][1:])
    values, counts = np.unique(z[:, 0], return_counts=True)

    if laplace_smoothing:
        token_set = np.unique(Y)
        entropies = np.array([entropy(np.concatenate([g, token_set])) for g in groups])
    else:
        entropies = np.array([entropy(g) for g in groups])
    # print(entropies)
    probs = counts / np.sum(counts)
    return np.sum(probs * entropies)


def better_than_random(Y: np.ndarray, X: np.ndarray, baseline: float = 0.5) -> bool:
    """Tests if the `information gain <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_
    of the input series given a conditioning series H(Y|X) is better than random.

    Args:
        Y: tokenized input 1D array. The entropy is calculated on this series.
        X: tokenized input 1D array. The conditioning array
        baseline: minimum amount of information gain for the ``Y`` array to be considered
            non-random.

    Returns:
        A True/False indicating whether information is exchanged between X and Y

        True means that it is random, False means that there is information exchanged
    """
    return information_gain(Y, X) > baseline


def information_gain(Y: np.ndarray, X: np.ndarray) -> float:
    """Calculate the `information gain
    <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_ of the input series given a
    conditioning series H(Y|X).

    Args:
        Y: tokenized input 1D array. The entropy is calculated on this series.
        X: tokenized input 1D array. The conditioning array

    Returns:
        The entropy minus a random baseline
    """
    return entropy(Y) - conditional_entropy(Y, X)


def reward_function(df: wi.WiDataFrame, lag: int, mask: bool = False) -> float:
    """Test for a reward function in the data H(r|state,action) based on their `conditional
    entropies <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Args:
        df: a discretized dataframe.
        lag: int for the lag.

    Returns:
        Returns the conditional entropy of reward given various lags. It is masked if the
        information gain is better than random
    """

    if mask:
        return better_than_random(
            tokenize(df[df.rewards]), tokenize(df[df.states + df.actions].shift(lag))
        )
    else:
        return information_gain(
            tokenize(df[df.rewards]), tokenize(df[df.states + df.actions].shift(lag))
        )


def stationary_policy(df: wi.WiDataFrame, lag: int, mask: bool = False) -> float:
    """Test for a stationary policy in the data H(action|state) based on their `conditional
    entropies <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Args:
        df: a discretized dataframe.
        lag: int for the lag.

    Returns:
        Returns the conditional entropy of action given various lags. It is masked if the
        information gain is better than random
    """

    if mask:
        return better_than_random(tokenize(df[df.actions]), tokenize(df[df.states].shift(lag)))
    else:
        return information_gain(tokenize(df[df.actions]), tokenize(df[df.states].shift(lag)))


def is_markovian(df: wi.WiDataFrame, lag: int, mask: bool = False) -> float:
    """Test for the Markov property in the data H(state|prev_state, prev_action) based on their
    `conditional entropies <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Args:
        df: a discretized dataframe.
        lag: int for the lag.

    Returns:
        Returns the conditional entropy of future stat given various lags. It is masked if the
        information gain is better than random
    """

    if mask:
        return better_than_random(
            tokenize(df[df.states]), tokenize(df[df.states + df.actions].shift(lag))
        )
    else:
        return information_gain(
            tokenize(df[df.states]), tokenize(df[df.states + df.actions].shift(lag))
        )


def action_reward(df: wi.WiDataFrame, lag: int, mask: bool = False) -> float:
    """Test for the effect of the action on the reward in the data H(reward|prev_action).

    Args:
        df: a discretized dataframe.
        lag: int for the lag.

    Returns:
        Returns the conditional entropy of future reward given various lags. It is masked if the
        information gain is better than random

    See Also
    --------
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)

    """
    if mask:
        return better_than_random(tokenize(df[df.rewards]), tokenize(df[df.actions].shift(lag)))
    else:
        return information_gain(tokenize(df[df.rewards]), tokenize(df[df.actions].shift(lag)))


def action_effective(df: wi.WiDataFrame, lag: int, mask: bool = False) -> float:
    """Test for the effect of the action on the state in the data H(state|prev_action) based on
    their `conditional entropies <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Args:
        df: a discretized dataframe.
        lag: int for the lag.

    Returns:
        Returns the conditional entropy of future states given various lags. It is masked if the
        information gain is better than random
    """

    if mask:
        return better_than_random(tokenize(df[df.states]), tokenize(df[df.actions].shift(lag)))
    else:
        return information_gain(tokenize(df[df.states]), tokenize(df[df.actions].shift(lag)))


def markovian_matrix(df: wi.WiDataFrame, lags: int = 10) -> pd.DataFrame:
    """Test for the key MDP properties based on their `conditional entropies
    <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Args:
        df: a discretized dataframe.
        lags: the number of distinct lags (0 to ``lags-1``) to test.
    Returns:
        df: dataframe with the results of various tests
    """

    data = [
        [
            "Markov Order f(state,action)=> next_state" if lag == 0 else is_markovian(df, lag)
            for lag in range(0, lags)
        ],
        [
            "Reward Function f(state,action)=> reward" if lag == 0 else reward_function(df, lag)
            for lag in range(0, lags)
        ],
        [
            "Action Contribution f(action)=> reward" if lag == 0 else action_reward(df, lag)
            for lag in range(0, lags)
        ],
        [
            "Action Effectiveness f(action)=> state" if lag == 0 else action_effective(df, lag)
            for lag in range(0, lags)
        ],
    ]
    labels = ["Test" if lag == 0 else "Lag_" + str(lag) for lag in range(0, lags)]

    mask = [
        [
            "Markov Order f(state,action)=> next_state" if lag == 0 else is_markovian(df, lag, True)
            for lag in range(0, lags)
        ],
        [
            "Reward Function f(state,action)=> reward"
            if lag == 0
            else reward_function(df, lag, True)
            for lag in range(0, lags)
        ],
        [
            "Action Contribution f(action)=> reward" if lag == 0 else action_reward(df, lag, True)
            for lag in range(0, lags)
        ],
        [
            "Action Effectiveness f(action)=> state"
            if lag == 0
            else action_effective(df, lag, True)
            for lag in range(0, lags)
        ],
    ]

    df = pd.DataFrame(data, columns=labels).set_index("Test")
    mask = pd.DataFrame(mask, columns=labels).set_index("Test")
    return df.mask(mask)


def normalized_markovian_matrix(df: wi.WiDataFrame) -> pd.DataFrame:
    """Test for the key MDP properties based on their `conditional entropies
    <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Args:
        df: a discretized dataframe.

    Returns:
        df: row normalized dataframe with the results of various tests
    """
    results = markovian_matrix(df)
    # return results.div(results.min(axis=1), axis=0)

    results = (
        results.sub(np.nanmin(results, axis=1), axis=0)
        .div(np.nanmax(results, axis=1) - np.nanmin(results, axis=1), axis=0)
        .fillna(0)
    )

    return results.where(results > 0.0, np.nan)


def data_generator_simple(
    markov_order: int = 0,
    action_effect: bool = False,
    policy: bool = False,
    reward_function: bool = False,
) -> wi.WiDataFrame:
    """Generate different types of data for your testing.

    Args:
        markov_order: the order of the synthetic data
        action_effect: allow the action to have an effect on the states
        policy: generate the actions with some rules
        reward_function: create a reward function with states and actions

    Reference:
        if markov_order=0 then the states are randomly generated
        if markov_order=1 then the next state is affected by the previous one only
        if markov_order>1 then the next state is affected by a mixture of the previous history.
        Keep this number less than 10.

        if action_affect = True then the actions can affect the state as well by
        using a different transition function

        if policy = True then there is a consistent rule choosing the action

        if reward_function = True then the reward function is calculated on the states

    """

    # Initial conditions
    state = np.array([[5, 5]])
    T1 = np.array([[0.7, 0.3], [0.6, 0.4]])
    T2 = np.array([[0.5, 0.5], [0.3, 0.7]])

    if markov_order == 0:

        wi_df = wi.WiDataFrame(
            pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("abcd")),
            states=["a", "b"],
            actions=["c"],
            rewards=["d"],
        )
        tokeniser = wi.DiscreteTokenizer(n_bins=50, num_bins_strategy="uniform")
        wi_df_tok = tokeniser.fit_transform(wi_df)

        return wi_df_tok

    else:
        df = pd.DataFrame()
        for i in range(1000):
            if policy:
                A = i % 5
            else:
                A = np.random.choice([0, 5])

            if markov_order == 1:
                if action_effect and (A % 2 != 1):
                    T = T2
                else:
                    T = T1
                next_I = state @ T
            else:
                if action_effect and (A % 2 != 1):
                    T = T2
                else:
                    T = T1
                next_I = (10 - markov_order) * (state @ T) / 10 + markov_order * state / 10

            if reward_function:
                R = next_I[0, 0]
            else:
                R = np.random.choice([0, 10])

            cur_states = state.reshape(-1)
            actions = np.array(A).reshape(-1)
            rewards = R

            temp_df = pd.DataFrame(
                {"s1": cur_states[0], "s2": cur_states[1], "a1": actions[0], "rewards": rewards},
                index=[0],
            )

            df = pd.concat([df, temp_df])

            state = next_I

        wi_df = wi.WiDataFrame(
            df,
            states=["s1", "s2"],
            actions=["a1"],
            rewards=["rewards"],
        )

        tokeniser = wi.DiscreteTokenizer(n_bins=50, num_bins_strategy="uniform")
        wi_df_tok = tokeniser.fit_transform(wi_df)

        return wi_df_tok


def data_generator_gym(
    env_name: str = "Taxi-v3",
    trainer: type[BaseAlgorithm] = A2C,
    training_steps: int = 10000,
    capture_steps: int = 1000,
) -> wi.WiDataFrame:
    """Generate a :class:`a2rl.WiDataFrame` from any well-defined OpenAi gym.

    An agent is trained first for ``training_steps``. Then, capture ``capture_steps`` from the
    trained agent.

    Args:
        env_name: Name of the gym environment.
        trainer: An underlying generator algorithm that supports discrete actions, such as
            :class:`stable_baselines3.dqn.DQN` or :class:`stable_baselines3.a2c.A2C`. Raise an error
            when passing a trainer that does not support discrete actions, such as
            :class:`stable_baselines3.sac.SAC`.
        training_steps: The number of steps to train the generator.
        capture_steps: The number of steps to capture.

    Returns:
        Whatif data frame.
    """
    # env_name = 'FrozenLake-v1'
    # env_name ='CartPole-v1'
    # env_name ='MountainCar-v0'
    # env_name = 'FrozenLake8x8-v1'
    # env_name= 'BipedalWalker-v3'

    env = gym.make(env_name)
    model = trainer(policy="MlpPolicy", env=env, verbose=False)  # type: ignore[call-arg,arg-type]
    model.learn(total_timesteps=training_steps)

    cap_env = wi.WhatifWrapper(env)
    model.set_env(cap_env)
    model.learn(total_timesteps=capture_steps)

    tokeniser = wi.DiscreteTokenizer(n_bins=50)
    df = tokeniser.fit_transform(cap_env.df)

    return df
