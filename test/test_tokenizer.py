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

from typing import List, Union

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import KBinsDiscretizer
from typeguard import check_type

import a2rl as wi
from a2rl.tokenizer import DiscreteTokenizer, compute_bases

# flake8: noqa


@pytest.fixture
def materials() -> tuple[wi.WiDataFrame, wi.WiDataFrame, wi.WiDataFrame]:
    # fmt: off
    arr_input = [
        [ 10,  5,  20, "x", "a"],
        [ 10,  5,  40, "x", "a"],
        [ 50,  5,  50, "y", "b"],
        [ 50, 85,  60, "y", "b"],
        [ 90, 85,  80, "z", "b"],
        [ 90, 85, 100, "z", "a"],
    ]

    arr_transformed = [
        [15, 17, 0, 5, 10],
        [15, 17, 1, 5, 10],
        [16, 18, 1, 5, 12],
        [16, 18, 2, 9, 12],
        [16, 19, 3, 9, 14],
        [15, 19, 4, 9, 14],
    ]

    arr_reconstructed = [
        ["a", "x", 28.0, 13.0, 18.0],
        ["a", "x", 44.0, 13.0, 18.0],
        ["b", "y", 44.0, 13.0, 50.0],
        ["b", "y", 60.0, 77.0, 50.0],
        ["b", "z", 76.0, 77.0, 82.0],
        ["a", "z", 92.0, 77.0, 82.0],
    ]
    # fmt: on

    sar_d = dict(states=list("ABC"), actions=["D"], rewards=["E"])
    df_input = wi.WiDataFrame(arr_input, columns=list("ABCDE"), **sar_d)
    df_transformed = wi.WiDataFrame(arr_transformed, columns=list("EDCBA"), **sar_d)
    df_reconstructed = wi.WiDataFrame(arr_reconstructed, columns=list("EDCBA"), **sar_d)
    return df_input, df_transformed, df_reconstructed


def test_compute_bases_wrong_input():
    with pytest.raises(
        ValueError,
        match="^Expect 1D array, but getting bins_per_column whose shape=.*",
    ):
        compute_bases(np.asarray([[1], [2]]))


@pytest.mark.parametrize(
    "input,expected",
    [
        ([], []),
        ([10], [0]),
        ([10, 2], [0, 10]),
        ([10, 2, 3], [0, 10, 12]),
        ([10, 2, 3, 4], [0, 10, 12, 15]),
    ],
)
def test_compute_bases_correct_input(input, expected):
    actual = compute_bases(bins_per_column=np.asarray(input)).tolist()
    assert actual == expected


def test_numeric_bins_strategy_uniform(materials):
    df = materials[0]
    DiscreteTokenizer().fit(df)


def test_numeric_bins_strategy_quantile(materials):
    df = materials[0]
    with pytest.warns(UserWarning, match="Bins whose width are too small"):
        DiscreteTokenizer(num_bins_strategy="quantile").fit(df)


def test_kbins_discretizer_nbins_invalid():
    a = np.asarray([range(200), range(200)]).T
    for i in range(2):
        with pytest.raises(
            ValueError,
            match=(
                "^KBinsDiscretizer received an invalid number of bins. "
                "Received .*, expected at least 2."
            ),
        ):
            KBinsDiscretizer(n_bins=i, encode="ordinal", strategy="uniform").fit(a)


@pytest.mark.parametrize("max_bins", (300,))
def test_kbins_discretizer_sweep_nbins(max_bins):
    a = np.asarray([range(200), range(200)]).T
    failed_n_bins = {}
    for i in range(2, max_bins):
        try:
            KBinsDiscretizer(n_bins=i, encode="ordinal", strategy="uniform").fit(a)
        except Exception as e:  # pragma: no cover
            failed_n_bins[i] = (e.__class__.__name__, str(e))
    print(f"{failed_n_bins=}")
    assert len(failed_n_bins) == 0


def test_fit_transform_separately(materials):
    df, expected = materials[0], materials[1]
    t = DiscreteTokenizer(n_bins=5, num_bins_strategy="uniform")
    reversed_cols = list(reversed(df.columns))
    t.fit(df[reversed_cols])
    actual = t.transform(df)

    assert "".join(actual.columns) == "EDCBA"
    assert (actual.values == expected.values).all()
    assert df.sar_d == expected.sar_d == actual.sar_d


def test_fit_transform_single_call(materials):
    df, expected = materials[0], materials[1]
    t = DiscreteTokenizer(n_bins=5, num_bins_strategy="uniform")
    reversed_cols = list(reversed(df.columns))
    actual = t.fit_transform(df[reversed_cols])

    assert "".join(actual.columns) == "EDCBA"
    assert (actual.values == expected.values).all()
    assert df.sar_d == expected.sar_d == actual.sar_d


def test_inverse_transform(materials):
    df, expected = materials[0], materials[2]
    t = DiscreteTokenizer(n_bins=5, num_bins_strategy="uniform")
    reversed_cols = list(reversed(df.columns))
    t.fit(df[reversed_cols])
    actual = t.inverse_transform(t.transform(df))

    assert "".join(actual.columns) == "EDCBA"
    assert np.issubdtype(actual.iloc[:, 2:].values.dtype, np.floating)
    assert np.allclose(expected.iloc[:, 2:].values, actual.iloc[:, 2:].values)
    assert (expected[list("ED")] == actual[list("ED")]).all(axis=None)
    assert df.sar_d == expected.sar_d == actual.sar_d


def test_fit_transform_cat_only_df(materials):
    df, expected = materials[0], materials[1]
    df = df[list("DE")]
    expected = expected[list("DE")]

    t = DiscreteTokenizer(n_bins=5)
    reversed_cols = list(reversed(df.columns))
    t.fit(df[reversed_cols])
    actual = t.transform(df)

    assert "".join(actual.columns) == "ED"
    assert df.sar_d == expected.sar_d == actual.sar_d


def test_fit_transform_num_only_df(materials):
    df, expected = materials[0], materials[1]
    df = df[list("ABC")]
    expected = expected[list("ABC")]

    t = DiscreteTokenizer(n_bins=5)
    reversed_cols = list(reversed(df.columns))
    t.fit(df[reversed_cols])
    actual = t.transform(df)

    assert "".join(actual.columns) == "CBA"
    assert df.sar_d == expected.sar_d == actual.sar_d


def test_valid_tokens():
    N = 100
    a = np.asarray([range(200), range(200)]).T
    df = pd.DataFrame(a, columns=["state_1", "state_2"])
    df["action"] = list("A" * N + "B" * N)
    t = DiscreteTokenizer()
    t.fit(df)

    expected = {"state_1": list(range(100)), "state_2": list(range(100, 200)), "action": [200, 201]}
    for i, (col_name, expected_tokens) in enumerate(expected.items()):
        for c in (i, col_name):
            actual = t.valid_tokens(c)
            check_type(f"actual_valid_tokens", actual, List[Union[int, np.integer]])
            assert expected_tokens == actual


def test_repr_and_str():
    t = DiscreteTokenizer(n_bins=5)
    assert str(t)
    assert repr(t)


def test_check_categorical_columns():
    wi.DiscreteTokenizer().check_categorical_columns(pd.DataFrame({"a": list("AB")}))

    with pytest.raises(ValueError):
        wi.DiscreteTokenizer().check_categorical_columns(pd.DataFrame({"a": list("AA")}))

    with pytest.raises(ValueError):
        wi.DiscreteTokenizer().check_categorical_columns(
            pd.DataFrame({"a": list("AA"), "b": list("BB"), "c": list("CD")})
        )


def test_check_numerical_columns():
    wi.DiscreteTokenizer().check_numerical_columns(pd.DataFrame({"a": [0, 1]}))

    with pytest.raises(ValueError):
        wi.DiscreteTokenizer().check_numerical_columns(pd.DataFrame({"a": [0, 0]}))

    for i in (None, np.nan, np.inf, pd.NA):
        with pytest.raises(ValueError):
            wi.DiscreteTokenizer().check_numerical_columns(pd.DataFrame({"a": [0, i]}))

        with pytest.raises(ValueError):
            wi.DiscreteTokenizer().check_numerical_columns(
                pd.DataFrame({"a": [0, i], "b": [0, i], "c": [0, 1]})
            )
