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
import gym
import pytest
from stable_baselines3 import DQN, SAC

import a2rl as wi
from a2rl.utils import data_generator_gym, data_generator_simple


def test_simple_generator():
    toy_data = data_generator_simple()
    assert isinstance(toy_data, wi.WiDataFrame)

    toy_data = data_generator_simple(
        markov_order=0,
        reward_function=True,
        action_effect=True,
        policy=True,
    )
    assert isinstance(toy_data, wi.WiDataFrame)

    toy_data = data_generator_simple(
        markov_order=1,
        reward_function=True,
        action_effect=False,
        policy=False,
    )
    assert isinstance(toy_data, wi.WiDataFrame)

    toy_data = data_generator_simple(
        markov_order=5,
        reward_function=True,
        action_effect=True,
        policy=True,
    )
    assert isinstance(toy_data, wi.WiDataFrame)

    toy_data = data_generator_simple(
        markov_order=5,
        reward_function=True,
        action_effect=True,
        policy=False,
    )
    assert isinstance(toy_data, wi.WiDataFrame)


def test_gym_generator():
    gym_data = data_generator_gym(env_name="Taxi-v3", trainer=DQN)
    assert isinstance(gym_data, wi.WiDataFrame)

    with pytest.raises(AssertionError, match=r"Discrete(.*) was provided"):
        gym_data = data_generator_gym(env_name="MountainCar-v0", trainer=SAC)


def test_wrapper_reset():
    env = wi.WhatifWrapper(gym.make("Taxi-v3"))

    i = env.reset()
    assert isinstance(i, int)

    i, d = env.reset(return_info=True)
    assert isinstance(i, int) and isinstance(d, dict)
