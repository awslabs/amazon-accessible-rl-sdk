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
"""Sample train script for Habana Gaudi on DL1 instance.

.. code-block:: console

    # Run on CPU, e.g., MBP M1.
    python notebooks/experimental/train-lightning.py --epochs 2

    # Run on DL1. NOTE: it's important to use python3.8 which contains habana frameworks.
    python3.8 notebooks/experimental/train-lightning.py --devices 1 --precision 16 --batch-size 512

    # Confirm that a python process is running on device 0
    hl-smi -i 0000:10:1d.0

    # Monitor hpu utilization
    hl-smi -i 0000:10:1d.0 -Q timestamp,index,utilization.aip -f csv -l 1

Open issues:
- During epoch 0, HPU doesn't seem to be utilized by roughly the 1st-half of the epoch. Why?
"""

# FIXME: Is it possible to make only rank 0 spits out log?

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from loguru import logger

import a2rl as wi
from a2rl.experimental.lightgpt import LightGPTBuilder, WarmupCosineLearningRateDecay


def parse_args() -> argparse.Namespace:
    default_train_config = str(Path(wi.experimental.lightgpt.__file__).parent / "config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-F",
        "--fast-mode",
        action="store_true",
        help="Use only a small fraction of dataset.",
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=int,
        default=-1,
        help="Number of devices (defaults: -1 which means use all)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=0,
        help="Override batch size to train with (0 to use the default in config.yaml)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=0,
        help="Override training epochs (0 to use the default in config.yaml)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="number of workers for dataloading",
    )
    parser.add_argument(
        "-c",
        "--train_config_file",
        default=default_train_config,
        help=f"YAML file of training config (defaults: {default_train_config})",
    )
    parser.add_argument(
        "-r",
        "--precision",
        type=int,
        default=32,
        help="fp precision to use, e.g. 32/16 (defaults: 32)",
    )
    parser.add_argument(
        "-o",
        "--default-root-dir",
        type=str,
        default=Path(__file__).parent / "model-lightgpt",
        help="best model checkpoint will be written at this location (defaults: .)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("CLI args = {}", vars(args))

    # Load data
    wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller"))
    if args.fast_mode:
        wi_df = wi_df.head(500)
    wi_df.add_value()
    tokenizer = wi.AutoTokenizer(wi_df, block_size_row=2)

    # Load training config
    with open(args.train_config_file) as f:
        train_config = yaml.safe_load(f)["train_config"]
        for k in ("epochs", "batch_size", "num_workers"):
            v = getattr(args, k, 0)
            if v > 0:
                train_config[k] = v

    logger.info("Training args = {}", train_config)

    # Setup trainer
    logger.info("Preparing the learning rate schedule")
    epoch_tokens = len(tokenizer.train_dataset)  # Number of tokens backpropped in 1x iteration
    lr_decay = WarmupCosineLearningRateDecay(
        learning_rate=6e-4,
        warmup_tokens=epoch_tokens // 2,
        final_tokens=train_config["epochs"] * epoch_tokens,
    )
    logger.info("Preparing the lightning trainer")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=("auto" if args.devices < 0 else args.devices),
        benchmark=False,
        max_epochs=train_config["epochs"],
        gradient_clip_val=1.0,
        callbacks=[lr_decay, pl.callbacks.ModelSummary(max_depth=2)],
        precision=args.precision,
        default_root_dir=args.default_root_dir,
    )

    wrapper = LightGPTBuilder(
        tokenizer,
        kw_args={"trainer": trainer},
        model_dir=args.default_root_dir,
        config=dict(train_config=train_config),
    )
    wrapper.fit()
