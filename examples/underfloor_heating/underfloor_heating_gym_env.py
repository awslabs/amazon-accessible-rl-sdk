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
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from gym.spaces import Box

import a2rl as wi


class WhatifWrapperUnderfloor(wi.WhatifWrapper):
    """This is data collector helper class.

    When agent is interacting with the env, it will store the states/actions/reward
    into a whatif dataframe.

    """

    def __init__(self, env: UnderfloorEnv):
        super().__init__(env)
        self.sar_d = {
            "states": env.state_cols,
            "actions": env.action_cols,
            "rewards": ["reward"],
        }
        self.cols = env.state_cols + env.action_cols + ["reward"]
        self.df = wi.WiDataFrame(
            pd.DataFrame(columns=self.cols, dtype="float"),
            **self.sar_d,
        )


def cal_reward(df: pd.DataFrame, quantile=0.5, optimal_room_temp=21) -> pd.DataFrame:
    """
    Calculate reward.

    The objective is to reduce the amount of deviation from optimal temperature, where
    the smaller the value the better.

    The return reward is the negetive of this amount, hence the larger the better.

    """
    over_temperature_cost = quantile * (df.room_temperature - optimal_room_temp)
    under_temperature_cost = (1 - quantile) * (optimal_room_temp - df.room_temperature)
    df["reward"] = -1 * np.where(
        over_temperature_cost > under_temperature_cost,
        over_temperature_cost,
        under_temperature_cost,
    )
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df.time.dt.hour
    df["week"] = df.time.dt.isocalendar().week

    df["sin_time"] = np.sin(2 * np.pi * df.hour / 24)
    df["cos_time"] = np.sin(2 * np.pi * df.hour / 24)

    df["sin_week"] = np.sin(2 * np.pi * df.week / 52)
    df["cos_week"] = np.cos(2 * np.pi * df.week / 52)
    return df


def get_data(data_path):
    df = pd.read_csv(data_path, parse_dates=["time"])
    return df


@dataclass
class UnderfloorEnvConfig:
    """Configuration class for Underfloor Gym Environment."""

    data_path: str | Path = Path("./") / "notebooks/underfloor_heating/underfloor_heating.csv"
    optimal_room_temp: float = 21
    quantile: float = 0.5
    noise: float = 0.1
    episode_len: int = 100
    debug: bool = False


class UnderfloorEnv(gym.Env):
    """
    This is a underfloow heating room environment. It uses historic weather and supply water
    temperature to predict the next room temp, given the action
    (underfloor temperature setpoint).

    A linear transition model is used to predict the next room temperature.

    States:
        - room_temperature
        - outside_temperature
        - outside_humidity
        - sin_time
        - cos_time
        - sin_week
        - cos_week

    Actions:
        - return_water_temp_setpoint

    Rewards:
        - Deviation from optimal comfort temperature.

    .. code-block:: python

        >>> env_config = UnderfloorEnvConfig(episode_len=1) # doctest:+SKIP
        >>> env = UnderfloorEnv(env_config) # doctest:+SKIP
        >>> state = env.reset() # doctest:+SKIP


    Linear model coefficient to predict next room temperature is coded inside variable
    transition_model_supply_water, with the following features:

        - supply_water_temp
        - room_temperature
        - outside_temperature
        - supply_water_temp
        - sin_time
        - cos_time
        - sin_week
        - cos_week
        - return_water_temp_setpoint

    """

    def __init__(self, env_config: UnderfloorEnvConfig):

        if not isinstance(env_config, UnderfloorEnvConfig):
            raise ValueError(f"Config must be of type UnderfloorEnvConfig, not {type(env_config)}")

        self.env_config = env_config
        self.transition_model_supply_water = np.array(
            [
                9.91735115e-01,
                2.41682941e-03,
                -4.06694354e-03,
                -5.35351747e-04,
                -5.35351747e-04,
                -1.29655058e-03,
                5.23551272e-06,
                1.01775251e-02,
            ]
        )

        self.transition_model = self.transition_model_supply_water

        self.min_floor_temp: float = 18
        self.max_floor_temp: float = 40
        self.min_obs: float = -50
        self.max_obs: float = 100

        df = get_data(self.env_config.data_path)
        create_features(df)

        self.state_cols = [
            "room_temperature",
            "outside_temperature",
            "supply_water_temp",
            "sin_time",
            "cos_time",
            "sin_week",
            "cos_week",
        ]
        self.action_cols = ["return_water_temp_setpoint"]
        self.reward_cols = ["reward"]

        self.df = df
        self.action_space = Box(
            self.min_floor_temp,
            self.max_floor_temp,
            shape=(len(self.action_cols),),
            dtype=np.float32,
        )
        self.observation_space = Box(
            self.min_obs,
            self.max_obs,
            shape=(len(self.state_cols),),
            dtype=np.float32,
        )

    def get_context(self):
        """Get context in the form of whatif dataframe."""

        self.sar_d = {
            "states": self.state_cols,
            "actions": self.action_cols,
            "rewards": self.reward_cols,
        }
        self.cols = self.state_cols + self.action_cols + self.reward_cols
        temp = np.concatenate((self.state, np.array([-99, -99]))).reshape(1, -1)

        self.context = wi.WiDataFrame(
            pd.DataFrame(temp, columns=self.cols, dtype="float"),
            **self.sar_d,
        )
        return self.context

    def reset(self, **kwargs) -> np.ndarray | tuple[np.ndarray, dict]:
        self.timestep = 0
        self.external_data = self._get_hist_external_data(self.timestep)
        self.room_temperature = self.df["room_temperature"].iloc[self.timestep]
        self.state = np.concatenate((self.room_temperature, self.external_data), axis=None)
        self.state = self.state.astype(np.float32)

        if self.env_config.debug:
            print("room temp", self.room_temperature)
            print(type(self.state), self.state)

        if kwargs.get("return_info", False):
            return self.state, dict()
        else:
            return self.state

    def step(self, action: list[int]) -> tuple[np.ndarray, float, bool, dict]:

        state_action = np.concatenate((self.state, action), axis=None)
        # print(f"{self.state=}")
        # print(f"{state_action=}")
        # assert len(state_action) == len(self.transition_model)

        self.room_temperature = np.dot(state_action, self.transition_model)
        # Calculate reward
        reward = cal_reward(pd.DataFrame(self.state.reshape(1, -1), columns=self.state_cols))[
            "reward"
        ][0]

        # Get next states
        self.timestep += 1
        self.external_data = self._get_hist_external_data(self.timestep)
        self.state = np.concatenate((self.room_temperature, self.external_data), axis=None)
        self.state = self.state.astype(np.float32)

        done = False
        if self.timestep >= self.env_config.episode_len:
            done = True

        return self.state, reward, done, {}

    def _get_hist_external_data(self, timestep: int) -> np.ndarray:
        """Return exogenous variables."""
        return (
            self.df[
                [
                    "outside_temperature",
                    "supply_water_temp",
                    "sin_time",
                    "cos_time",
                    "sin_week",
                    "cos_week",
                ]
            ]
            .iloc[timestep]
            .values
        )


if __name__ == "__main__":
    env_config = UnderfloorEnvConfig(episode_len=5, data_path="underfloor_heating.csv")
    env = UnderfloorEnv(env_config)
    state = env.reset()
