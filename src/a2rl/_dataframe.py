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

import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Literal, Protocol

import gym
import numpy as np
import pandas as pd
from typing_extensions import TypeGuard

import a2rl as wi

from ._io import Metadata, save_metadata


# TODO
def _get_constructor(klass, example: HasSarAttributes, *args, **kwargs) -> Callable:
    def _constructor(*args, **kwargs):
        return klass(
            *args,
            states=example.states,
            actions=example.actions,
            rewards=example.rewards,
            **kwargs,
        )

    return _constructor


# TODO
def _sar_from(o: HasSarAttributes):
    return o.states, o.actions, o.rewards


def _pre_to_csv(
    o: WiDataFrame | WiSeries,
    path_or_buf: str | PathLike[str],
    forced_categories: None | list[str] = None,
    compact: bool = False,
) -> Path:
    """Create output directory and metadata.

    Args:
        path_or_buf: The path name of the output dir.

    Returns:
        Path: path to output directory.
    """
    p = path_or_buf if isinstance(path_or_buf, Path) else Path(path_or_buf)
    if p.exists():
        warnings.warn(f"Existing directory {p} will be overwritten.")
    p.mkdir(parents=True, exist_ok=True)
    save_metadata(
        Metadata(
            states=o.states,
            actions=o.actions,
            rewards=o.rewards,
            forced_categories=forced_categories,
        ),
        p / "metadata.yaml",
        compact=compact,
    )
    return p


class HasSarAttributes(Protocol):
    @property
    def states(self) -> list[str]:
        raise NotImplementedError

    @property
    def actions(self) -> list[str]:
        raise NotImplementedError

    @property
    def rewards(self) -> list[str]:
        raise NotImplementedError


class SarMixin:
    # Declarations for mypy
    _states: list[str]  #: Expected state colum names (list[str])
    _actions: list[str]  #: Expected action column names (list[str])
    _rewards: list[str]  #: Expected rewards column names (list[str])

    @property
    def sar_d(self) -> dict[str, list[str]]:
        """The dictionary of 585 expected *sar* column names.

        Returns:
            ``{'states': [str], 'actions': [str], 'rewards': [str]}``.

        See Also
        --------
        sar : The list of the expected *sar* columns.
        states : The expected state columns.
        actions : The expected action columns.
        rewards : The expected reward columns.
        """
        return dict(
            states=list(self._states),
            actions=list(self._actions),
            rewards=list(self._rewards),
        )

    @property
    def sar(self) -> list[str]:
        """The list of the expected *sar* column names.

        Returns:
            The expected *sar* column names, in the order of states, actions, and reward.

        See Also
        --------
        sar_d : The dictionary of the expected *sar* columns.
        states : The expected state columns.
        actions : The expected action columns.
        rewards : The expected reward columns.
        """
        return self.states + self.actions + self.rewards

    def _set_sar(self, **kwargs) -> None:
        """Set ``self._{k} = {v}`` for each ``k:v`` in ``kwargs``."""
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, f"_{k}", list(v))

    @property
    def states(self) -> list[str]:
        """The list of the expected column names of states.

        Returns:
            The expected column names for states.

        See Also
        --------
        sar : The list of the expected *sar* columns.
        sar_d : The dictionary of the expected *sar* columns.
        actions : The expected action columns.
        rewards : The expected reward columns.
        """
        return list(self._states)

    @property
    def actions(self) -> list[str]:
        """The list of the expected column names of actions.

        Returns:
            The expected column names for actions.

        See Also
        --------
        sar : The list of the expected *sar* columns.
        sar_d : The dictionary of the expected *sar* columns.
        states : The expected state columns.
        rewards : The expected reward columns.
        """
        return list(self._actions)

    @property
    def rewards(self) -> list[str]:
        """The list of the expected column names of rewards.

        Returns:
            The expected column names for rewards.

        See Also
        --------
        sar : The list of the expected *sar* columns.
        sar_d : The dictionary of the expected *sar* columns.
        states : The expected state columns.
        actions : The expected action columns.
        """
        return list(self._rewards)


class WiSeries(pd.Series, SarMixin):
    _metadata = ["_states", "_actions", "_rewards"]

    def __init__(
        self,
        data=None,
        states: None | Collection[str] = None,
        actions: None | Collection[str] = None,
        rewards: None | Collection[str] = None,
        **kwargs,
    ) -> None:
        """A ``WiSeries`` object is a :class:`pandas.Series` with additional metadata on the
        expected column names for ``states``, ``actions``, and ``rewards`` (i.e., the *sar*
        columns).

        In addition to the standard :class:`pandas.Series` constructor arguments, a ``WiSeries``
        also accepts the following keyword arguments:

        Args:
            states: The expected column names for states.
            actions: The expected column names for actions.
            rewards: The expected column names for rewards.

        .. warning::
            This class is mainly used internally by ``whatif``.

            By design, the name of a ``WiSeries`` is equal to zero or one expected *sar* column
            name.

        See Also
        --------
        WiDataFrame
        pandas.Series


        Examples:
            Create a new ``WiSeries``:

            .. code-block:: python

                >>> import a2rl as wi
                >>> ser = wi.WiSeries(
                ...     [11, 12, 13],
                ...     name="s0",
                ...     states=["s1", "s2"],
                ...     actions=["a"],
                ...     rewards=["r"],
                ... )

                >>> ser
                0    11
                1    12
                2    13
                Name: s0, dtype: int64

                >>> ser.sar
                ['s1', 's2', 'a', 'r']

            Inherit *sar* columns from the source ``WiDataFrame``:

            .. code-block:: python

                >>> df = wi.WiDataFrame(
                ...     {
                ...         "s": [0, 1, 2],
                ...         "a": ["x", "y", "z"],
                ...         "r": [0.5, 1.5, 2.5],
                ...     },
                ...     states=["s"],
                ...     actions=["a"],
                ...     rewards=["r"],
                ... )
                >>> ser = df['a']
                >>> ser.sar
                ['s', 'a', 'r']
        """
        if rewards and len(rewards) > 2:
            raise ValueError(f"rewards can have at most two columns, but received {rewards}")
        super().__init__(data=data, **kwargs)
        if isinstance(data, SarMixin) and [states, actions, rewards] == [None] * 3:
            states, actions, rewards = _sar_from(data)

        self._set_sar(states=states, actions=actions, rewards=rewards)

    @property
    def _constructor(self):
        return _get_constructor(WiSeries, self)

    @property
    def _constructor_expanddim(self):
        _c_e = _get_constructor(WiDataFrame, self)

        # See:
        # https://github.com/geopandas/geopandas/blob/51864acf3dd0bcbc74b2a922c6e012d7e57e46b5/geopandas/geoseries.py#L66-L69
        #
        #     pd.concat (pandas/core/reshape/concat.py) requires this for the
        #     concatenation of series since pandas 1.1
        #     (https://github.com/pandas-dev/pandas/commit/f9e4c8c84bcef987973f2624cc2932394c171c8c)
        #
        # E.g., required by df.groupby().agg({'a': 'min', 'b': 'max'})
        _c_e._get_axis_number = WiDataFrame._get_axis_number

        return _c_e

    @property
    def _values(self) -> np.ndarray:
        # https://github.com/pandas-dev/pandas/issues/46554#issuecomment-1084305476
        return super()._values

    def to_csv_dataset(
        self,
        path_or_buf: str | PathLike[str],
        *args,
        forced_categories: None | Iterable[str] = None,
        compact: bool = False,
        **kwargs,
    ) -> None:
        """Save this series as a ``Whatif`` dataset.

        This method has similar signatures to :meth:`pandas.Series.to_csv()`, however with some
        changes.

        Args:
            path_or_buf: Unlike :meth:`pandas.Series.to_csv()`, this accepts only path name of
                the output dir.
            args: passed to :meth:`pandas.Series.to_csv()`.
            compact: When True, do not write ``None`` entries to the output metadata YAML.
            kwargs: passed to :meth:`pandas.Series.to_csv()`.
        """
        if not (forced_categories is None or isinstance(forced_categories, list)):
            forced_categories = list(forced_categories)

        outdir = _pre_to_csv(
            self,
            path_or_buf,
            forced_categories=forced_categories,
            compact=compact,
        )
        self.to_csv(outdir / "data.csv", *args, **kwargs)


class WiDataFrame(pd.DataFrame, SarMixin):
    _metadata = ["_states", "_actions", "_rewards"]

    def __init__(
        self,
        data=None,
        states: None | Collection[str] = None,
        actions: None | Collection[str] = None,
        rewards: None | Collection[str] = None,
        **kwargs,
    ) -> None:
        """A ``WiDataFrame`` object is a :class:`pandas.DataFrame` with additional metadata on the
        expected column names for ``states``, ``actions``, and ``rewards`` (i.e., the *sar*
        columns).

        In addition to the standard :class:`pandas.DataFrame` constructor arguments, a
        ``WiDataFrame`` also accepts the following keyword arguments:

        Args:
            states: The expected column names for states.
            actions: The expected column names for actions.
            rewards: The expected column names for rewards.

        .. note::
            By design, a ``WiDataFrame`` itself may miss one or more of the *sar* columns.
            Downstream tasks should deal with missing *sar* columns.

            Some downstream tasks such as slicing ignores the discrepancy, while RL-related tasks
            may require all *sar* columns presented.

        See Also
        --------
        WiSeries
        pandas.DataFrame


        Examples
        --------
        Create a new ``WiDataFrame``:

        .. code-block:: python

            >>> import a2rl as wi
            >>> df = wi.WiDataFrame(
            ...     {
            ...         "s1": [1, 2, 3],
            ...         "s2": [3, 4, 5],
            ...         "sess": [0, 0, 0],
            ...         "z": [6, 7, 8],
            ...         "a": ["x", "y", "z"],
            ...         "r": [0.5, 1.5, 2.5],
            ...     },
            ...     states=["s1", "s2"],
            ...     actions=["a"],
            ...     rewards=["r"],
            ... )

            >>> df
               s1  s2  sess  z  a    r
            0   1   3     0  6  x  0.5
            1   2   4     0  7  y  1.5
            2   3   5     0  8  z  2.5

        Check the metadata:

        .. code-block:: python

            >>> df.sar
            ['s1', 's2', 'a', 'r']

            >>> df.sar_d
            {'states': ['s1', 's2'], 'actions': ['a'], 'rewards': ['r']}

            >>> df.states
            ['s1', 's2']

            >>> df.actions
            ['a']

            >>> df.rewards
            ['r']

        Slice the states. The resulted ``WiDataFrame`` or ``WiSeries`` inherits the expected
        *sar* columns from the source ``DataFrame``.

        .. code-block:: python

            >>> df[df.states]
               s1  s2
            0   1   3
            1   2   4
            2   3   5

            >>> df[df.states].sar
            ['s1', 's2', 'a', 'r']

        Take just the *sar* columns:

        .. code-block:: python

            >>> df.trim()
               s1  s2  a    r
            0   1   3  x  0.5
            1   2   4  y  1.5
            2   3   5  z  2.5
        """
        if rewards and len(rewards) > 2:
            raise ValueError(f"rewards can have at most two columns, but received {rewards}")
        super().__init__(data=data, **kwargs)
        if isinstance(data, SarMixin) and [states, actions, rewards] == [None] * 3:
            states, actions, rewards = _sar_from(data)
        self._set_sar(states=states, actions=actions, rewards=rewards)

    def trim(self, copy: bool = False) -> WiDataFrame:
        """Get the *sar* columns of this data frame.

        Raise an error when any of the expected *sar* column names is missing from this data frame.

        Args:
            copy: True to return a new copy of data frame, False to return a view to this data
                frame.

        Returns:
            Data frame with only the *sar* columns. If ``copy=False``, the returned data frame is a
            a view to this data frame, else a new data frame.
        """
        view = self[self.states + self.actions + self.rewards]
        return view if not copy else view.copy()

    @property
    def _constructor(self):
        return _get_constructor(WiDataFrame, self)

    @property
    def _constructor_sliced(self):
        return _get_constructor(WiSeries, self)

    def to_csv_dataset(
        self,
        path_or_buf: str | PathLike[str],
        *args,
        forced_categories: None | Iterable[str] = None,
        compact: bool = False,
        **kwargs,
    ) -> None:
        """Save this data frame as a ``Whatif`` dataset.

        This method has similar signatures to :meth:`pandas.DataFrame.to_csv()`, however with some
        changes.

        Args:
            path_or_buf: Unlike :meth:`pandas.DataFrame.to_csv()`, this accepts only path name of
                the output dir.
            args: passed to :meth:`pandas.DataFrame.to_csv()`.
            kwargs: passed to :meth:`pandas.DataFrame.to_csv()`.

        See Also
        --------
        read_csv_dataset


        Example:

            Save a ``WiDataFrame`` to directory ``/tmp/my-dataset``.

            .. code-block:: python

                >>> from a2rl import WiDataFrame
                >>> df = WiDataFrame(
                ...     {
                ...         "i": [3, 4, 5],
                ...         "s": [1, 2, 3],
                ...         "j": [4, 5, 6],
                ...         "a": ["x", "y", "z"],
                ...         "k": ["z", "x", "y"],
                ...         "r": [0.5, 1.5, 2.5],
                ...     },
                ...     states=["s"],
                ...     actions=["a"],
                ...     rewards=["r"],
                ... )

                >>> df
                   i  s  j  a  k    r
                0  3  1  4  x  z  0.5
                1  4  2  5  y  x  1.5
                2  5  3  6  z  y  2.5

                >>> df.to_csv_dataset("/tmp/my-dataset")
        """
        if not (forced_categories is None or isinstance(forced_categories, list)):
            forced_categories = list(forced_categories)
        outdir = _pre_to_csv(
            self,
            path_or_buf,
            forced_categories=forced_categories,
            compact=compact,
        )
        self.to_csv(outdir / "data.csv", *args, **kwargs)

    @property
    def sequence(self) -> np.ndarray:
        """Return a 1D Numpy representation of the DataFrame, in row-major order.

        Returns:
            The sequence of the data frame.

        Example:

            .. code-block:: python

                >>> from a2rl import WiDataFrame
                >>> df = WiDataFrame(
                ...     {
                ...         "sess": [0, 0, 0],
                ...         "s": [1, 2, 3],
                ...         "a": ["x", "y", "z"],
                ...         "r": [0.5, 1.5, 2.5]
                ...     },
                ...     states=["s"],
                ...     actions=["a"],
                ...     rewards=["r"],
                ... )

                >>> df
                   sess  s  a    r
                0     0  1  x  0.5
                1     0  2  y  1.5
                2     0  3  z  2.5

                >>> df.sequence
                array([0, 1, 'x', 0.5, 0, 2, 'y', 1.5, 0, 3, 'z', 2.5], dtype=object)
        """
        return self.values.ravel()

    def _check_add_value_args(
        self,
        value_col: str,
        override: Literal["replace", "warn", "error"],
        alpha: float,
        gamma: float,
    ) -> None:
        if not getattr(self, "_rewards", None):
            raise ValueError(f"Unspecified reward column: {getattr(self, '_rewards', None)}")

        if value_col == self._rewards[0]:
            raise ValueError(f"value_col={value_col} conflicts with reward")

        if value_col in self.columns:
            if override == "error":
                raise ValueError(f"Column {value_col} already exists in this WiDataFrame")
            elif override == "warn":
                warnings.warn(f"Column {value_col} will be overwritten")
            elif override != "replace":
                raise ValueError(f"Unknown override: {override}")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Learning rate alpha={alpha} not in 0 and 1.")

        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Discount factor gamma={gamma} not in 0 and 1.")

    def add_value(
        self: WiDataFrame,
        alpha: float = 0.1,
        gamma: float = 0.6,
        sarsa: bool = True,
        value_col: str = "value",
        override: Literal["replace", "warn", "error"] = "replace",
    ) -> WiDataFrame:
        """Append column ``value_col`` into this dataframe (restriction: ``df`` must NOT contain
        column names ``_state``, ``_action``, ``_reward``, and the ``value_col``).

        Args:
            alpha: Learning rate in `Q-Learning <https://en.wikipedia.org/wiki/Q-learning>`_ and
                `SARSA <https://en.wikipedia.org/wiki/State-action-reward-state-action>`_. Must be
                be within 0 and 1.
            gamma: Discount factor of future reward in Q-Learning and SARSA. Must be within 0 and 1.
            sarsa: When ``True``, compute the value using the `SARSA Bellman equation
                <https://en.wikipedia.org/wiki/State-action-reward-state-action>`_ which is a
                conservative on-policy temporal difference update. When ``False``, use the
                `Q-Learning Bellman equation <https://en.wikipedia.org/wiki/Q-learning>`_ which is
                an off-policy temporal difference update.
            value_col: The column name for the computed values.
            override: What to do when this dataframe has had column ``value_col``. Valid values
                are ``replace`` to silently override, ``warn`` to show a warning, and ``raise`` to
                raise a :exc:`ValueError`.

        Returns:
            This dataframe, modified with an additional ``value_col`` column. This return value is
            provided to facilitate chaining as-per the functional programming style.
        """
        self._check_add_value_args(value_col, override, alpha, gamma)
        if len(self._rewards) == 1:
            df = self
        else:
            df = WiDataFrame(
                self,
                states=self.states,
                actions=self.actions,
                rewards=self.rewards[:1],
            )
        df_t = wi.DiscreteTokenizer(n_bins=50).fit_transform(df.trim())

        # Temp df with only three columns: _state, _action, _reward
        df_t = pd.concat(  # type: ignore[assignment]
            [
                df_t[df_t.states].astype(str).apply("_".join, axis=1).astype("category").cat.codes,
                df_t[df_t.actions].astype(str).apply("_".join, axis=1).astype("category").cat.codes,
                df[df.rewards].reset_index(drop=True),
            ],
            axis=1,
            copy=False,
        )
        df_t.columns = ["_state", "_action", "_reward"]  # type: ignore[assignment]

        q_table = np.zeros([df_t["_state"].nunique(), df_t["_action"].nunique()])
        iterations = 10
        for n in range(iterations):
            for i in range(0, len(df_t) - 1):
                state = int(df_t.loc[i, "_state"])
                next_state = int(df_t.loc[i + 1, "_state"])
                action = int(df_t.loc[i, "_action"])
                reward = df_t.loc[i + 1, "_reward"]
                old_value = q_table[state, action]

                if sarsa:
                    next_value = q_table[next_state, np.argmax(q_table[next_state])]
                    new_value = (1 - alpha) * old_value + alpha * (
                        reward + gamma * next_value - old_value
                    )
                else:
                    next_max = np.max(q_table[next_state])
                    new_value = (1 - alpha) * old_value + alpha * (
                        reward + gamma * next_max - old_value
                    )

                q_table[state, action] = new_value

        self[value_col] = pd.Series(
            q_table[df_t["_state"].astype(int), df_t["_action"].astype(int)],
            index=self.index,
        )
        if len(self._rewards) < 2:
            self._rewards.append(value_col)
        else:
            self._rewards[1] = value_col
        return self


def is_old_gym_step(
    t: tuple[Any, Any, Any, Any] | tuple[Any, Any, Any, Any, Any]
) -> TypeGuard[tuple[Any, Any, Any, Any]]:
    """Determines whether tuple ``t`` is returned by ``gym<0.25.0``.

    Related commit: https://github.com/openai/gym/commit/907b1b20dd9ac0cba5803225059b9c6673702467.

    Args:
        t (tuple[Any, Any, Any, Any] | tuple[Any, Any, Any, Any, Any]): the return value of
            :method:~`gym.Env.step()`.

    Returns:
        TypeGuard[tuple[Any, Any, Any, Any]]: True if tuple ``t`` is returned by gym<0.25.0``
    """
    return len(t) == 4


def is_new_gym_step(
    t: tuple[Any, Any, Any, Any] | tuple[Any, Any, Any, Any, Any]
) -> TypeGuard[tuple[Any, Any, Any, Any, Any]]:
    """Determines whether tuple ``t`` is returned by ``gym>=0.25.0``.

    Related commit: https://github.com/openai/gym/commit/907b1b20dd9ac0cba5803225059b9c6673702467.

    Args:
        t (tuple[Any, Any, Any, Any] | tuple[Any, Any, Any, Any, Any]): the return value of
            :method:~`gym.Env.step()`.

    Returns:
        TypeGuard[tuple[Any, Any, Any, Any, Any]]: True if tuple ``t`` is returned by
            ``gym>=0.25.0``.
    """
    return len(t) == 5


class WhatifWrapper(gym.Wrapper[Any, np.ndarray]):
    """Record the transitions in the OpenAI gym :class:`gym.Env` into a Whatif data frame.

    Args:
        env: a gym environment.

    Examples
    --------

    .. code-block:: python

        >>> import gym
        >>> import a2rl as wi
        >>> from stable_baselines3 import DQN
        >>> from stable_baselines3.common.evaluation import evaluate_policy
        >>> from stable_baselines3.ppo import MlpPolicy
        >>>
        >>> env_name = "Taxi-v3"
        >>> env = gym.make(env_name)
        >>> model = DQN(policy="MlpPolicy", env=env, verbose=False)
        >>> model.learn(total_timesteps=10)  # doctest: +SKIP
        >>> eval_env = gym.make(env_name)
        >>> eval_env = wi.WhatifWrapper(eval_env)
        >>> mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        >>>
        >>> eval_env.df.info()  # doctest: +NORMALIZE_WHITESPACE
        <class 'a2rl._dataframe.WiDataFrame'>
        Int64Index: 2000 entries, 0 to 0
        Data columns (total 3 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   0       2000 non-null   float64
         1   1       2000 non-null   float64
         2   2       2000 non-null   float64
        dtypes: float64(3)
        memory usage: ... KB
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.episode = 0

        state = env.observation_space.sample()
        state = np.array(state).ravel()
        state_length = state.size

        action = env.action_space.sample()
        action = np.array(action).ravel()
        action_length = action.size

        self.sar_d = {
            "states": np.arange(state.size),
            "actions": np.arange(action.size) + state.size,
            "rewards": [action.size + state.size],
        }
        self.df = WiDataFrame(
            pd.DataFrame(columns=np.arange(state_length + action_length + 1), dtype="float"),
            **self.sar_d,
        )

        self._state: Any

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, dict]:
        """Wrapper to :func:`gym.Wrapper.step()` which records one timestep of the environment's
        dynamics.

        Args:
            action (object): an action provided by the agent
        """
        step_results = self.env.step(action)
        if is_old_gym_step(step_results):
            # gym<0.24.0
            next_state, reward, done, info = step_results
        elif is_new_gym_step(step_results):
            # gym>=0.25.0.
            # See: https://github.com/openai/gym/commit/907b1b20dd9ac0cba5803225059b9c6673702467
            next_state, reward, done, _, info = step_results
        else:
            raise ValueError(f"Invalid tuple length: len(step_results)={len(step_results)}")

        action = np.array(action).ravel()
        stacked = np.hstack(
            [
                np.array(self._state).ravel(),
                np.array(action).ravel(),
                np.array(reward),
            ]
        )
        self.df = pd.concat(  # type: ignore[assignment]
            [
                self.df,
                WiDataFrame(stacked.reshape(1, -1), columns=list(self.df), **self.sar_d),
            ]
        )

        self._state = next_state
        return next_state, reward, done, info

    def reset(self, **kwargs) -> tuple[gym.core.ObsType, dict] | gym.core.ObsType:
        """Wrapper to :func:`gym.Wrapper.reset()` which resets the environment to an initial state
        and returns an initial observation.

        Returns:
            observation: Observation of the initial state.
            info (optional dictionary): returned only when ``return_info=True``.
        """
        self.episode += 1
        reset_result = self.env.reset(**kwargs)
        if kwargs.get("return_info", False):
            obs = reset_result[0]
        else:
            obs = reset_result  # type: ignore[assignment]
        self._state = obs
        return reset_result
