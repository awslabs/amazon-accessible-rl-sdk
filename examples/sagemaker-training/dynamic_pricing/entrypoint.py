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
"""

.. code-block:: console

    # Assuming pwd is GITROOT
    python examples/sagemaker-training/dynamic_pricing/entrypoint.py
"""
import argparse
import os
import pickle  # nosec: B403
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
from flight_sales import flight_sales_gym
from loguru import logger
from stable_baselines3 import A2C

import a2rl as wi

warnings.filterwarnings("ignore")


class Timer:
    def __init__(self) -> None:
        self.t_start = perf_counter()

    def reset(self) -> None:
        self.t_start = perf_counter()

    def log(self, msg: str, reset=True) -> None:
        t_end = perf_counter()
        logger.info(msg, t_end - self.t_start)
        if reset:
            self.t_start = t_end


# This helper function converts gym state space into tokens
def history_2_context(tokenizer, context_widf):
    custom_context = tokenizer.field_tokenizer.transform(context_widf).values.ravel()[
        : -len(context_widf.actions) - len(context_widf.rewards)
    ]

    return custom_context


# Here our agent returns the action with best p90 reward
def agent(simulator, ctx):
    recommendation_df = simulator.sample(ctx, max_size=500, as_token=False)
    return recommendation_df.groupby("ticket_price")["revenue"].quantile(0.9).idxmax()


if __name__ == "__main__":
    default_model_dir = Path(os.environ.get("SM_MODEL_DIR", "."))
    default_output_dir = default = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "."))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=default_model_dir,
        type=Path,
        help=f"Where to save model. Auto-detected default is {default_model_dir}",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        type=Path,
        help=f"Where to save miscellaneous output. Auto-detected default is {default_output_dir}",
    )
    parser.add_argument(
        "--quick-mode", default=0, type=int, help="Reduce number of steps for the gym"
    )
    args = parser.parse_args()
    logger.info(vars(args))

    env = flight_sales_gym()
    if args.quick_mode:
        gym_timesteps = (1000, 1000)
        eval_steps = 2
    else:
        gym_timesteps = (1000, 10000)
        eval_steps = 100
    logger.info(f"gym_timesteps = {gym_timesteps}")
    logger.info(f"eval_steps = {eval_steps}")

    # Here were are both generating offline data and training an RL agent at the same time. The
    # wrapper stores the trajectory information.
    trainer = A2C
    model = trainer(policy="MlpPolicy", env=env, verbose=False)  # type: ignore[call-arg,arg-type]
    model.learn(total_timesteps=gym_timesteps[0])

    cap_env = wi.WhatifWrapper(env)
    model.set_env(cap_env)
    model.learn(total_timesteps=gym_timesteps[1])

    wi_df = wi.WiDataFrame(
        cap_env.df.values,
        states=["season", "freight_price"],
        actions=["ticket_price"],
        rewards=["revenue"],
    )
    wi_df.columns = ["season", "freight_price", "ticket_price", "revenue"]  # type: ignore

    # Here we save both the tokerizer and the model
    field_tokenizer = wi.DiscreteTokenizer(num_bins_strategy="quantile")
    tokenizer = wi.AutoTokenizer(wi_df, field_tokenizer=field_tokenizer, block_size_row=2)
    model_dir = args.model_dir
    builder = wi.GPTBuilder(tokenizer, model_dir)
    if args.quick_mode:
        builder._configs["epochs"] = min(2, builder._configs["epochs"])
        logger.info("Running in quick-mode forces epochs to at most 2")
    builder.fit()  # Will also save the model to a .pt file

    # Remember to also save the tokenizer
    auto_tokenizer_pickle = model_dir / "auto-tokenizer.pickle"
    with open(auto_tokenizer_pickle, "wb") as f:  # type: ignore
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)  # type: ignore
    # Make sure can re-read auto-tokenizer
    with open(auto_tokenizer_pickle, "rb") as f:  # type: ignore
        tok2 = pickle.load(f)  # type: ignore  # nosec: B301

    timer = Timer()  # Evaluation is slow, so let's be verbose about timings.

    # Lets see how well the learned offline policy works
    logger.info("Lets see how well the learned offline policy works")
    env.reset()
    env.step(0.5)
    done = False
    ctx = history_2_context(tokenizer, env.context())
    # Here we use the tokenizer and the model for our policy
    simulator = wi.Simulator(tokenizer, builder.model)
    timer.reset()
    for i in range(eval_steps):
        action = agent(simulator, ctx)
        state, reward, done, msg = env.step(action)
        ctx = history_2_context(tokenizer, env.context())
    a2rl_reward = env.reward_history
    timer.log(
        "Generated rewards using a2rl simulator in {{:.1f}} seconds "
        f"for {eval_steps} trajectories"
    )

    # Lets compare it with the agent trained on the same data
    logger.info("Lets compare it with the agent trained on the same data")
    state = env.reset()
    done = False
    for i in range(eval_steps):
        action = model.predict(state)
        state, reward, done, msg = env.step(float(action[0]))
    rl_agent_reward = env.reward_history
    t_end = perf_counter()
    timer.log(
        "Generated rewards using stable_baseline3 in {{:.1f}} seconds "
        f"for {eval_steps} trajectories"
    )

    # save the plot
    matplotlib.use("Agg")
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(a2rl_reward, "g.", label="A2RL")
    ax.plot(rl_agent_reward, "rx", label="Trained Agent")
    ax.legend(loc="upper left")
    fig.savefig(args.output_dir / "results.png")
    timer.log("Finished saving plots in {:1.1f} seconds")
    # Free the plot.
    fig.clf()
    plt.close(fig)
    del fig
    fig = None
