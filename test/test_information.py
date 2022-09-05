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
#from a2rl.utils import NotMDPDataError, assert_mdp #, conditional_entropy, entropy
from a2rl.information import *

method = 'placebo-norm'


def test_random_walk():
    """Information must be exchanged between the previous state and the next state from the
    definition of a first order Markov Chain
    """
    state_changes = np.random.choice([-1,1], 1000, p=[0.5, 0.5])
    random_walk = np.cumsum(state_changes)
    X = random_walk[1:]
    A = random_walk[:-1]
    assert len(X) == len(A)

    gain, passed = conditional_information_test(X, A, method =method)
    assert passed

def test_random_walk_no_information():
    """
    No information exchanged between two random series
    """
    state_changes = np.random.choice([-1,1], 1000, p=[0.5, 0.5])

    gain, passed = conditional_information_test(state_changes, state_changes, method =method)
    assert not passed

def test_related_series_difference():
    
    low_entropy_token_set = np.random.randint(1, 10, size=1000)
    related_entropy_token_set = np.diff(np.hstack((0, low_entropy_token_set)))

    gain, passed = conditional_information_test(low_entropy_token_set,related_entropy_token_set, method =method)
    print(gain)
    assert passed

def test_related_series_sum():
    
    low_entropy_token_set = np.random.randint(1, 10, size=1000)
    related_entropy_token_set = low_entropy_token_set + 2

    gain, passed = conditional_information_test(low_entropy_token_set,related_entropy_token_set, method =method)
    print(gain)
    assert passed


def test_unrelated_series():
    
    low_entropy_token_set = np.random.randint(1, 5, size=100)
    unrelated_low_entropy_token_set = np.random.randint(1, 5, size=100)


    gain, passed = conditional_information_test(low_entropy_token_set, unrelated_low_entropy_token_set, method =method)
    print(gain)
    assert not passed




"""
def test_base_entropy():
    assert entropy(low_entropy_token_set) < entropy(high_entropy_token_set)
    assert conditional_entropy(
        related_entropy_token_set,
        low_entropy_token_set,
        
    ) < entropy(related_entropy_token_set)




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
"""