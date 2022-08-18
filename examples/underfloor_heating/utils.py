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
import numpy as np
import pandas as pd

import a2rl as wi


def obs_to_wi_context(tokenizer, obs: np.ndarray) -> wi.WiDataFrame:
    """Convert gym obs into whatif dataframe context.

    Assumption:
    The length of observation align with tokenizer state columns.

    """
    # print("*******obs_to_wi_context*********")
    # print("Input", obs, len(obs), sep="\n")
    if len(obs) != len(tokenizer.state_columns):
        raise ValueError(f"Invalid obs length, expecting {len(tokenizer.state_columns)}")

    DUMMY_VAL = 0

    sar_d = {
        "states": tokenizer.state_columns,
        "actions": tokenizer.action_columns,
        "rewards": tokenizer.reward_columns,
    }
    all_cols = tokenizer.state_columns + tokenizer.action_columns + tokenizer.reward_columns
    dummy_cols = np.array([DUMMY_VAL] * (tokenizer.action_dim + tokenizer.reward_dim))
    temp = np.concatenate((obs, dummy_cols)).reshape(1, -1)

    context = wi.WiDataFrame(pd.DataFrame(temp, columns=all_cols, dtype="float"), **sar_d)
    return context


def wi_context_to_token(tokenizer, context: wi.WiDataFrame) -> np.ndarray:
    """
    Convert whatif dataframe context into sequence of dataframe tokens,
    and then return the context without actions and rewards,
    so the output can be used for whatif ``sample`` api.

    """
    # print("*******wi_context_to_token*********")
    # print("Input", context, sep="\n")
    custom_context = tokenizer.field_tokenizer.transform(context).values.ravel()[
        -tokenizer.block_size : -tokenizer.action_dim - tokenizer.reward_dim
    ]
    # print("Output")
    # print(len(custom_context))
    # display(custom_context)
    return custom_context


def create_new_context(
    tokenizer, initial_context: wi.WiDataFrame, action: float, reward: float, obs: np.ndarray
) -> wi.WiDataFrame:
    """
    Create new context by appending the initial context (original value, non-token) with new action
    and observation from simulator.

    The maximum number of row return is based on tokenizer.
    """
    # print("*******create_new_context*******")
    # display(initial_context)
    # print(f"{action=}, {reward=}, {obs=}")
    temp = initial_context.values.ravel()

    # Concatenate initial context with action, rewards, next states
    last_action_idx = -tokenizer.action_dim - tokenizer.reward_dim
    last_reward_idx = -tokenizer.reward_dim
    temp[last_action_idx] = action
    temp[last_reward_idx] = reward
    # Added last dummy action/rewarda
    DUMMY_VAL = 0
    dummy_cols = np.array([DUMMY_VAL] * (tokenizer.action_dim + tokenizer.reward_dim))
    temp = np.concatenate((temp, obs, dummy_cols), axis=None)
    # Reshape back to dataframe shape
    temp = temp.reshape(-1, len(initial_context.columns))
    sar_d = {
        "states": tokenizer.state_columns,
        "actions": tokenizer.action_columns,
        "rewards": tokenizer.reward_columns,
    }
    all_cols = tokenizer.state_columns + tokenizer.action_columns + tokenizer.reward_columns

    context = wi.WiDataFrame(pd.DataFrame(temp, columns=all_cols, dtype="float"), **sar_d)
    # Truncated to max context rows
    context = context.iloc[-tokenizer.block_size_row :]

    # display(context)
    # print(f"{context.shape}")
    return context


def get_next_action(simulator: wi.Simulator, context: np.ndarray, max_size: int):
    """
    Get the next best action from a list of samples based on given context.
    """
    recommendation_df = simulator.sample(context, max_size=max_size, as_token=False)
    action = recommendation_df.iloc[recommendation_df.reward.idxmax(), 0]
    return action
