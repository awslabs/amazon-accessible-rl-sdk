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
"""Submit a job to train a A2RL simulator on SageMaker training.

.. code-block:: bash

    # Train with default settings
    python3 submit_train.py

    # Use this when not submitting from a SageMaker notebook instance (or Studio notebook).
    python3 submit_train \
        --role arn:aws:iam::111122223333:role/my-sagemaker-execution-role

    # Use custom config.yaml
    python3 submit_train.py \
        --role arn:aws:iam::111122223333:role/my-sagemaker-execution-role \
        --config-yaml dynamic_pricing/config.yaml

    # Handy for dev: verify what goes into the training code tarball starting a training job.
    python3 submit_train --generate-tarball-only

    # Submit a really short training job (~5min).
    # NOTE: CLI args after --config_yaml <...> are sent as hyperparameters directly to the
    #       entrypoint script. In this example, "--quick-mode 1" are passed as-is to
    #       dynamic_pricing/entrypoint.py.
    python3 submit_train \
        --role arn:aws:iam::111122223333:role/my-sagemaker-execution-role \
        --config-yaml dynamic_pricing/config.yaml \
        --quick-mode 1
"""
from __future__ import annotations

import smepu

import argparse
import sys
from pathlib import Path

from loguru import logger
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session


def get_defaults():
    defaults = argparse.Namespace

    # Change me!
    defaults.sagemaker_s3_prefix = f"s3://{Session().default_bucket()}/a2rl/sagemaker-training-jobs"

    try:
        import sagemaker

        defaults.role = sagemaker.get_execution_role()
    except ImportError:
        defaults.role = ""

    try:
        import a2rl as wi

        defaults.config_yaml = str(Path(wi.__file__).parent / "config.yaml")
        defaults.a2rl = str(Path(wi.__file__).parent)
    except ImportError:
        defaults.config_yaml = "config.yaml"
        defaults.a2rl = "a2rl"

    defaults.source_dir = Path(__file__).parent / "dynamic_pricing"

    return defaults


def get_parser(defaults) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser("sagemaker-train-submitter")
    parser.add_argument(
        "--role",
        default=defaults.role,
        help=f"SageMaker execution role. Auto-detected default is {defaults.role}",
    )
    parser.add_argument(
        "--config-yaml",
        default=defaults.config_yaml,
        help=(
            "Location of config.yaml to include in the code tarball. "
            f"Auto-detected default is {defaults.config_yaml}"
        ),
    )
    parser.add_argument(
        "--a2rl-location",
        default=defaults.a2rl,
        help=f"Parent folder of a2rl/ package. Auto-detected default is {defaults.a2rl}",
    )
    parser.add_argument(
        "--source-dir",
        default=defaults.source_dir,
        help=("SageMaker source_dir. " f"Auto-detected default is {defaults.source_dir}"),
    )
    parser.add_argument(
        "--generate-tarball-only",
        action="store_true",
        help="Generate sourcedir.tar.gz on Amazon S3, then exit.",
    )

    args, script_args = parser.parse_known_args()
    if not args.role:
        raise ValueError("Failed to autodetect SageMaker execution role. Please specify one.")

    return args, script_args


if __name__ == "__main__":
    defaults = get_defaults()
    args, script_args = get_parser(defaults)
    logger.info("CLI args: {}", vars(args))
    logger.info("Entrypoint args: {}", smepu.argparse.to_kwargs(script_args))

    # Meta-programming fun :) -- send remaining CLI args to entrypoint script (via hyperparameters).
    hyperparameters = smepu.argparse.to_kwargs(script_args)

    estimator = PyTorch(
        base_job_name="wi-dyn-pricing",
        entry_point="entrypoint.py",
        source_dir=str(args.source_dir),
        dependencies=[args.config_yaml, args.a2rl],
        framework_version="1.10",
        py_version="py38",
        code_location=defaults.sagemaker_s3_prefix,
        output_path=defaults.sagemaker_s3_prefix,
        max_run=24 * 60 * 60,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",  # NOTE: may need to raise limit on this account.
        # instance_type="ml.g4dn.12xlarge",  # NOTE: need to raise limit on this account.
        # instance_type="local_gpu",
        # instance_type="local",
        role=args.role,
        hyperparameters=hyperparameters,
        metric_definitions=[
            {"Name": "train:loss", "Regex": r"epoch\[\d+\] iter \[\d+\]: train loss (\S+)\. "},
            {"Name": "test:loss", "Regex": r"test loss: (\S+)"},
        ],
    )

    # For debugging: generage sourcedir.tar.gz on Amazon S3, without starting a training job.
    if args.generate_tarball_only:
        estimator._prepare_for_training()
        code_dir = estimator.uploaded_code.s3_prefix
        script = estimator.uploaded_code.script_name
        logger.success("(code_dir, script) = ({}, {})", code_dir, script)
        logger.info(
            (
                "Preview content: "
                "aws s3 cp {} - | tar --to-stdout -xzf - config.yaml requirements.txt"
            ),
            code_dir,
        )
        logger.info("List content: aws s3 cp {} - | tar -tzf -", code_dir)
        sys.exit(1)

    estimator.fit(wait=False)
    job_name = estimator._current_job_name
    logger.success("Training job name: {}", job_name)
    logger.info(
        "Describe training job: aws sagemaker describe-training-job --training-job-name {} | jq",
        job_name,
    )
    logger.info(
        "Model will be saved at: {}",
        f"{defaults.sagemaker_s3_prefix}/output/model.tar.gz",
    )
    logger.info(
        "Training output (incl. backtesting results) will be saved at: {}",
        f"{defaults.sagemaker_s3_prefix}/{job_name}/output/output.tar.gz",
    )
    logger.info(
        "URL: https://{}.console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}",
        estimator.sagemaker_session.boto_region_name,
        estimator.sagemaker_session.boto_region_name,
        job_name,
    )
