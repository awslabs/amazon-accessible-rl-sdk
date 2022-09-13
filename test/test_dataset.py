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

from unittest import mock

import pytest
from pandas.api.types import is_numeric_dtype

import a2rl as wi


def test_list_sample_datasets():
    assert wi.list_sample_datasets() == ["chiller", "rtu"]


@pytest.mark.parametrize(
    "dataset, expected_shape, expected_sar_d",
    (
        # Expected answers for the Chiller dataset
        (
            "chiller",
            (9153, 5),
            {
                "states": ["condenser_inlet_temp", "evaporator_heat_load_rt"],
                "actions": ["staging"],
                "rewards": ["system_power_consumption"],
            },
        ),
        # Expected answers for the RTU dataset
        (
            "rtu",
            (4335, 8),
            {
                "states": [
                    "outside_humidity",
                    "outside_temperature",
                    "return_humidity",
                    "return_temperature",
                ],
                "actions": ["economizer_enthalpy_setpoint", "economizer_temperature_setpoint"],
                "rewards": ["power"],
            },
        ),
    ),
)
def test_sample_dataset_path(dataset, expected_shape, expected_sar_d):
    dirname = wi.sample_dataset_path(dataset)
    df = wi.read_csv_dataset(dirname)
    assert df.sar_d == expected_sar_d
    assert df.shape == expected_shape


@mock.patch("a2rl.DiscreteTokenizer.check_numerical_columns")
@mock.patch("a2rl.DiscreteTokenizer.check_categorical_columns")
@mock.patch("a2rl.utils.assert_mdp")
def test_read_csv_dataset_test_mdp(mock_cnc, mock_ccc, mock_assert_mdp):
    # This test only ensures that the only accepted exception is NotMDPDataError.
    # Any other errors are considered as bug in the implementation.
    wi.read_csv_dataset(wi.sample_dataset_path("chiller"), test_mdp=True)
    mock_assert_mdp.assert_called()


@pytest.fixture
def df() -> wi.WiDataFrame:
    return wi.WiDataFrame(
        {
            "a": [0, 0, 1, 1, 2],
            "b": [1, 2, 3, 4, 5],
            "c": [2, 3, 4, 5, 6],
            "d": [3, 4, 5, 6, 7],
            "e": [4, 5, 6, 7, 8],
            "f": [5, 6, 7, 8, 9],
        },
        states=["a", "b"],
        actions=["c"],
        rewards=["d"],
    )


@pytest.mark.parametrize(
    "forced_categories,expected_types",
    (
        (["a", "b"], [False, False, True, True, True, True]),
        (None, [True] * 6),
    ),
)
def test_forced_categories(df, tmp_path, forced_categories, expected_types):
    metadata = wi.Metadata(**df.sar_d, forced_categories=forced_categories)
    p = tmp_path / f"mydataset-{wi.utils.timestamp()}"
    p.mkdir()
    wi.save_metadata(metadata, p / "metadata.yaml", compact=True)
    df.to_csv(p / "data.csv", index=False)

    df2 = wi.read_csv_dataset(p)
    is_numeric_series = [is_numeric_dtype(ser) for _, ser in df2.iteritems()]
    assert is_numeric_series == expected_types
