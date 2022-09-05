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

################################################################################
# Workaround to force silence deprecation warnings from select modules.
################################################################################
import warnings as _warnings

original_warn = _warnings.warn


def _warn(message, category=None, stacklevel=1, source=None):
    """Manually silence these warnings, which in some cases do not get silenced by -Wignore."""
    stacklevel += 1
    if category is not DeprecationWarning:
        original_warn(message, category, stacklevel, source)

    if message.startswith("distutils Version classes are deprecated."):
        # https://github.com/mwaskom/seaborn/issues/2724
        return
    if message.endswith("descriptors from generated code or query the descriptor_pool."):
        # https://github.com/tensorflow/tensorboard/issues/5798
        return

    # Still show any other deprecation warning.
    original_warn(message, category, stacklevel, source)


_warnings.warn = _warn
################################################################################

# flake8: noqa: E402
from . import utils
from ._dataframe import WhatifWrapper, WiDataFrame, WiSeries
from ._io import (
    Metadata,
    list_sample_datasets,
    read_csv_dataset,
    read_metadata,
    sample_dataset_path,
    save_metadata,
)
from ._metadatadict import MetadataDict
from .simulator import AutoTokenizer, GPTBuilder, Simulator, SimulatorDataset, SimulatorWrapper
from .tokenizer import DiscreteTokenizer, Tokenizer

__version__ = "1.0.1"
