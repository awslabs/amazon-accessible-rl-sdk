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
from pathlib import Path

import pytest


@pytest.mark.skip_slow
@pytest.mark.script_launch_mode("subprocess")
def test_train_lightgpt_script(script_runner, tmp_path):
    cmd = [
        str(Path(__file__).parent / "train-lightgpt.py"),
        "--fast-mode",
        "--epochs",
        "1",
        "--default-root-dir",
        str(tmp_path),
    ]
    ret = script_runner.run(*cmd)
    assert ret.success
