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

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt

import a2rl as wi

wi.utils.set_seed(42)
warnings.filterwarnings("ignore", category=UserWarning, module="a2rl.simulator")
plt.rcParams["figure.dpi"] = 200
plt.ioff()

# fmt: off
test_custom_context = np.array([
     26,  45, 352, 153, 259,  15,  83, 347, 145, 339,  20,  60, 343,  # noqa: E241, E131, E126
    154, 270,  26,  82, 352, 170, 271,  42,  79, 343, 151, 333,  22,  # noqa: E241, E131, E126
    111, 344, 182, 258,  17, 104, 350, 155, 289,  21, 107, 345, 156,  # noqa: E241, E131, E126
    325,  24,  71, 343, 158, 243,  11,  79, 347, 143, 281,  35,  84,  # noqa: E241, E131, E126
    343, 143, 320,  17, 113, 353, 212, 246,  28,  68, 353, 152, 300,  # noqa: E241, E131, E126
     18, 125, 352, 146, 272,  14,  96, 347, 144, 310,  32, 111, 351,  # noqa: E241, E131, E126
    208, 247,  39, 118, 352, 145, 336,  16, 122, 347, 149, 305,  21,  # noqa: E241, E131, E126
    107, 344, 143, 325,  12,  51, 347, 155, 243,  35, 113, 346, 143,  # noqa: E241, E131, E126
    252,  12, 119, 348, 198, 249,  12,  73, 345, 156, 266,  29,  98,  # noqa: E241, E131, E126
    348, 190, 290,  29, 109, 348, 143, 327,  29, 129, 348, 149, 264,  # noqa: E241, E131, E126
     17,  64, 353, 179, 299,  31,  63, 348, 162, 281,  35, 100, 344,  # noqa: E241, E131, E126
    148, 263,   5, 141, 343, 163, 244,  18,  61, 351, 200, 262,   1,  # noqa: E241, E131, E126
     61, 345, 147, 322,  31,  46, 347, 152, 301,  17, 128, 347, 189,  # noqa: E241, E131, E126
    301,  13,  87, 353, 145, 245,  20, 122, 347, 146, 248,  33, 127,  # noqa: E241, E131, E126
    350, 152, 309,  33, 126, 343, 153, 288,  37,  49, 348, 147, 308,  # noqa: E241, E131, E126
     16,  84, 345, 158, 299,  20,  80, 347, 155, 265,  13, 131, 348,  # noqa: E241, E131, E126
    184, 294,  40, 128, 350, 173, 310,  12,  81, 352, 198, 256,  25,  # noqa: E241, E131, E126
     63, 347, 188, 255,  39,  80, 345, 176, 282,   1, 140, 353, 174,  # noqa: E241, E131, E126
    252,  11,  99, 352, 168, 261,  27,  55, 351, 146, 311,  27, 112,  # noqa: E241, E131, E126
    348, 201, 260,  30,  89,                                          # noqa: E241, E131, E126
])  # noqa: E241, E131, E126
# fmt: on

MODEL_DIR = Path(__file__).parent / f"model-{Path(__file__).stem}"


def get_simulator(
    df_rows: int,
    epochs: int,
    reuse_model: bool,
    block_size_row: int = 2,
    model_dir: Path = MODEL_DIR,
) -> wi.Simulator:
    wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller"))
    wi_df.add_value()
    wi_df = wi_df.iloc[:df_rows]  # Speed up training for demo purpose

    # Instantiate a tokenier given the selected dataset.
    tokenizer = wi.AutoTokenizer(wi_df, block_size_row=block_size_row)
    builder = wi.GPTBuilder(tokenizer, model_dir)
    builder._configs["epochs"] = epochs  # Yet another hack to fix the training epochs :(

    # Any better way than hardcoding model.pt?
    if reuse_model and (Path(model_dir) / "model.pt").is_file():
        logger.info(f"Will load the GPT model from {model_dir}/")
        builder.load_model()
    else:
        logger.info("Training the GPT model")
        builder.fit()

    return wi.Simulator(tokenizer, builder.model)


class BmkResults:
    def __init__(self):
        self.func: list[str] = []
        self.ctx_horizon: list[int] = []
        self.batch_size: list[int] = []
        self.max_size_per_step: list[int] = []
        self.dur_secs: list[float] = []

    def append(
        self,
        *,
        func: str,
        ctx_horizon: int,
        batch_size: int,
        max_size_per_step: int,
        dur_secs: float,
    ) -> None:
        self.func.append(func)
        self.ctx_horizon.append(ctx_horizon)
        self.batch_size.append(batch_size)
        self.max_size_per_step.append(max_size_per_step)
        self.dur_secs.append(dur_secs)

    @property
    def df(self):
        return pd.DataFrame(
            {
                "func": self.func,
                "ctx_horizon": self.ctx_horizon,
                "batch_size": self.batch_size,
                "max_size_per_step": self.max_size_per_step,
                "dur_secs": self.dur_secs,
            }
        )


def bmk_lookahead(
    simulator: wi.Simulator,
    batch_sizes: list[int],
    max_size_per_step: int,
    ctx_horizon: int,
    results: BmkResults | None = None,
) -> BmkResults:
    tokenizer = simulator.tokenizer
    step_size = tokenizer.state_dim + tokenizer.action_dim + tokenizer.reward_dim
    max_ctx_horizon = (len(test_custom_context) - tokenizer.state_dim) // step_size
    if ctx_horizon > max_ctx_horizon:
        raise ValueError(f"Maximum ctx_horizon is {max_ctx_horizon}, but {ctx_horizon} is provided")

    custom_context = test_custom_context[: tokenizer.state_dim + step_size * ctx_horizon]
    # print("len(custom_context) =", len(custom_context))
    # print("custom_context.ndim =", custom_context.ndim)

    block_size = tokenizer.block_size
    if len(custom_context) > block_size:
        trunc_custom_context = custom_context[-block_size:]
    else:
        trunc_custom_context = custom_context
    action_cols = tokenizer.df.actions
    nb_actions = len(tokenizer.df_tokenized[action_cols[0]].unique())
    if max_size_per_step > nb_actions:
        raise ValueError(
            "bmk_lookahead() doesn't know how to proceed when max_action_size > nb_actions."
        )
    all_valid_actions = simulator.get_valid_actions(
        trunc_custom_context, max_size=max_size_per_step
    ).values

    if not results:
        results = BmkResults()

    for batch_size in batch_sizes:
        # Benchmark simulator.batch_lookahead()
        batch_custom_context = np.tile(custom_context, (batch_size, 1))
        stt = time.time()
        reward, next_states = simulator.lookahead(batch_custom_context, all_valid_actions)
        dur = time.time() - stt
        logger.debug(f"BATCH_LOOKAHEAD: batch_size={batch_size}, dur_secs={dur}")
        results.append(
            func="batch_lookahead",
            ctx_horizon=ctx_horizon,
            batch_size=batch_size,
            max_size_per_step=max_size_per_step,
            dur_secs=dur,
        )
        # print(reward.shape, next_states.shape)

        # Benchmark simulator.lookahead()
        stt = time.time()
        for i in range(batch_size):
            for act in all_valid_actions:
                actl = act.tolist()
                reward, next_states = simulator.lookahead(custom_context, actl)
        dur = time.time() - stt
        logger.debug(f"      LOOKAHEAD: batch_size={batch_size}, dur_secs={dur}")
        results.append(
            func="lookahead",
            ctx_horizon=ctx_horizon,
            batch_size=batch_size,
            max_size_per_step=max_size_per_step,
            dur_secs=dur,
        )

    return results


def bmk_sample(
    simulator: wi.Simulator,
    batch_sizes: list[int],
    max_size_per_step: int,
    ctx_horizon: int,
    results: BmkResults | None = None,
) -> BmkResults:
    tokenizer = simulator.tokenizer
    step_size = tokenizer.state_dim + tokenizer.action_dim + tokenizer.reward_dim
    max_ctx_horizon = (len(test_custom_context) - tokenizer.state_dim) // step_size
    if ctx_horizon > max_ctx_horizon:
        raise ValueError(f"Maximum ctx_horizon is {max_ctx_horizon}, but {ctx_horizon} is provided")

    custom_context = test_custom_context[: tokenizer.state_dim + step_size * ctx_horizon]

    # custom_context = tokenizer.df_tokenized.iloc[0, :2].values
    # array([26, 45])

    per_ctx_max_size = max_size_per_step

    if not results:
        results = BmkResults()

    for batch_size in batch_sizes:
        # Benchmark simulator.batch_sample()
        batch_custom_context = np.tile(custom_context, (batch_size, 1))
        # print(custom_context)
        # print("custom_context.ndim = ", custom_context.ndim)
        stt = time.time()
        recommendation_df = simulator.sample(
            batch_custom_context, max_size=per_ctx_max_size, as_token=True
        )
        dur = time.time() - stt
        logger.debug(f"BATCH_SAMPLE: batch_size={batch_size}, dur_secs={dur}")
        results.append(
            func="batch_sample",
            ctx_horizon=ctx_horizon,
            batch_size=batch_size,
            max_size_per_step=max_size_per_step,
            dur_secs=dur,
        )
        # print(recommendation_df.shape)

        # Benchmark simulator.sample()
        # print(custom_context)
        # print("custom_context.ndim = ", custom_context.ndim)
        stt = time.time()
        for _ in range(batch_size):
            recommendation_df = simulator.sample(  # noqa: F841
                custom_context, max_size=per_ctx_max_size, as_token=True
            )
        dur = time.time() - stt
        logger.debug(f"      SAMPLE: batch_size={batch_size}, dur_secs={dur}")
        results.append(
            func="sample",
            ctx_horizon=ctx_horizon,
            batch_size=batch_size,
            max_size_per_step=max_size_per_step,
            dur_secs=dur,
        )

    return results


def parse_args(model_dir: Path = MODEL_DIR):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_sizes",
        metavar="N",
        default="[10, 20, 40]",
        type=json.loads,
        help=(
            "Batch sizes, loop sizes, or nb_runs. Must be a JSON list of integers. "
            "Remember to single-quote to prevent the JSON string from getting interpreted by your "
            "shell. Default: [10, 20, 40]"
        ),
    )

    parser.add_argument(
        "--ctx_horizon",
        type=int,
        default=2,
        help=(
            "The step count in the input context to Simulator.lookahead(). Internally, this "
            "script flattens CTX_HORIZON steps (i.e., dataframe rows) into the corresponding "
            "sequence of (s, a, r, ..., s). [Default: 2]"
        ),
    )

    parser.add_argument(
        "--max_size_per_step",
        type=int,
        default=2,
        help=(
            "Given context (s, a, r, ..., s): (i) call Simulator.lookahead() with "
            "MAX_SIZE_PER_STEP actions to obtain MAX_SIZE_PER_STEP rewards, and (ii) call "
            "Simulator.sample(..., max_size=MAX_SIZE_PER_TIMESTEP) to obtain MAX_SIZE_PER_TIMESTEP "
            "(a, r). [Default: 2]"
        ),
    )

    parser.add_argument(
        "--reuse_model",
        action="store_true",
        help=f"Load model from {str(MODEL_DIR)}/ if it exists. [Default: False]",
    )

    parser.add_argument(
        "--no-plot_speedup",
        action="store_false",
        default=True,
        dest="plot_speedup",
        help="Do not plot speed up sub-plot [Default: plot speed-up]",
    )

    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of training epochs. [Default: 1]",
    )

    parser.add_argument(
        "--df_rows",
        default=300,
        type=int,
        help="Number of dataframe rows. [Default: 300]",
    )

    parser.add_argument(
        "--plot_subtitle",
        default=None,
        help="Subtitle of the output plot. [Default: nothing]",
    )

    args = parser.parse_args()

    if [i for i in args.batch_sizes if i < 1]:
        raise ValueError(f"Non-positive batch size in {args.batch_sizes}")

    if args.ctx_horizon < 1:
        raise ValueError(f"Non-positive CTX_HORIZON: {args.ctx_horizon}")

    if args.max_size_per_step < 1:
        raise ValueError(f"Non-positive MAX_SIZE_PER_STEP: {args.max_size_per_step}")

    if args.epochs < 1:
        raise ValueError(f"Non-positive epochs: {args.epochs}")

    if args.df_rows < 300:
        raise ValueError(f"Minimum 300 rows, but instead got {args.args.df_rows} ")

    return args


def save_plot(df: pd.DataFrame, path: Path, title: str = "", plot_speedup: bool = False) -> None:
    if plot_speedup:
        fig, two_ax = plt.subplots(2, sharex=True)
        first_ax = two_ax[0]
    else:
        first_ax = None
    ax = sns.lineplot(
        data=df,
        x="batch_size",
        y="dur_secs",
        hue="func",
        style="func",
        markers=True,
        linewidth=0.5,
        ax=first_ax,
    )
    ax.grid(ls="--", alpha=0.5)
    if plot_speedup:
        t1 = df[1::2].dur_secs  # sample
        t2 = df[0::2].dur_secs  # batch_sample
        speed_up = [tt1 / tt2 for (tt1, tt2) in zip(t1, t2)]
        bszlist = df.batch_size[::2]
        funclist = df.func[::2]
        funclist = [x + " speedup" for x in funclist]
        df_spd = pd.DataFrame({"batch_size": bszlist, "speedup": speed_up, "func": funclist})

        ax1 = sns.lineplot(
            data=df_spd,
            x="batch_size",
            y="speedup",
            hue="func",
            style="func",
            markers=True,
            linewidth=0.5,
            ax=two_ax[1],
        )
        ax1.grid(ls="--", alpha=0.5)

    if title:
        ax.set_title(title)
    if not plot_speedup:
        fig = ax.figure
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "bmk-results.png")

    # Whatever possible ways to release figure
    fig.clf()
    plt.close(fig)
    del fig


if __name__ == "__main__":
    args = parse_args()
    logger.info("CLI args = {}", vars(args))
    simulator = get_simulator(args.df_rows, args.epochs, args.reuse_model)

    results = bmk_lookahead(
        simulator,
        args.batch_sizes,
        args.max_size_per_step,
        args.ctx_horizon,
    )

    bmk_sample(
        simulator,
        args.batch_sizes,
        args.max_size_per_step,
        args.ctx_horizon,
        results=results,
    )

    df = results.df
    logger.debug("Benchmark results:\n{}", df)
    df.to_csv(MODEL_DIR / "bmk-results.csv", index=False)

    title = f"ctx_horizon={args.ctx_horizon}, max_size_per_step={args.max_size_per_step}"
    if args.plot_subtitle:
        title = "\n".join([title, args.plot_subtitle])
    save_plot(df, MODEL_DIR / "bmk-results.png", title=title, plot_speedup=args.plot_speedup)

    logger.success("Benchmark completed")
