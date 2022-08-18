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
import pytest

import a2rl as wi
from a2rl.utils import NotMDPDataError, assert_mdp, conditional_entropy, entropy

low_entropy_token_set = np.array([1, 3, 5, 2, 3, 5, 3, 2, 1, 3, 4, 5, 1, 1, 2, 3, 4])
related_entropy_token_set = np.diff(np.hstack((0, low_entropy_token_set)))
unrelated_entropy_token_set = np.random.randint(1, 5, size=len(low_entropy_token_set))
high_entropy_token_set = np.arange(len(low_entropy_token_set))


def test_base_entropy_laplace_smoothing():
    assert entropy(low_entropy_token_set) < entropy(high_entropy_token_set)

    print(f"{conditional_entropy(related_entropy_token_set, unrelated_entropy_token_set)=}")
    print(f"{entropy(related_entropy_token_set)=}")
    diff = conditional_entropy(related_entropy_token_set, unrelated_entropy_token_set) - entropy(
        related_entropy_token_set
    )
    print(f"{diff=}")
    assert abs(diff) < 0.5  # Not adding information, hence delta is fixed to at most half-bit.


def test_base_entropy():
    assert entropy(low_entropy_token_set) < entropy(high_entropy_token_set)
    assert conditional_entropy(
        related_entropy_token_set,
        low_entropy_token_set,
        laplace_smoothing=False,
    ) < entropy(related_entropy_token_set)


def test_base_entropy_no_laplace_smoothing():
    assert entropy(low_entropy_token_set) < entropy(high_entropy_token_set)

    print(f"{conditional_entropy(related_entropy_token_set, unrelated_entropy_token_set)=}")
    print(f"{entropy(related_entropy_token_set)=}")
    diff = conditional_entropy(related_entropy_token_set, unrelated_entropy_token_set) - entropy(
        related_entropy_token_set
    )
    print(f"{diff=}")
    assert abs(diff) < 0.5  # Not adding information, hence delta is fixed to at most half-bit.


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            0,
        ),
        (
            np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            1,
        ),
    ],
)
def test_entropy(data, expected):
    result = entropy(data)
    print(f"{result=}")
    assert result == expected


def test_assert_mdp():
    # This test only ensures that the only accepted exception is NotMDPDataError.
    # Any other errors are considered as bug in the implementation.
    for dataset in ("chiller", "rtu"):
        try:
            # Aggressively speed-up the test using small number of groups and lags.
            df_tok = wi.DiscreteTokenizer(n_bins=5).fit_transform(
                df=wi.read_csv_dataset(wi.sample_dataset_path(dataset)),
                check=False,
            )
            assert_mdp(df_tok, lags=2)
        except NotMDPDataError:
            pass
