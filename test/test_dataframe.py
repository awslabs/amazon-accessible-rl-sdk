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
import pandas as pd
import pytest
import yaml
from pandas.api.types import is_numeric_dtype, is_object_dtype

import a2rl as wi


@pytest.fixture
def sar_d():
    return dict(states=["a", "b"], actions=["c"], rewards=["d"])


@pytest.fixture
def sar(sar_d):
    return sar_d["states"] + sar_d["actions"] + sar_d["rewards"]


@pytest.fixture
def data():
    return {
        "a": [0, 0, 1, 1, 2],
        "b": [1, 2, 3, 4, 5],
        "c": [2, 3, 4, 5, 6],
        "d": [3, 4, 5, 6, 7],
        "e": [4, 5, 6, 7, 8],
        "f": [5, 6, 7, 8, 9],
    }


@pytest.fixture
def df(data, sar_d):
    return wi.WiDataFrame(data, **sar_d)


def test_from_pd_df(data, sar_d):
    df = pd.DataFrame(data)
    df2 = wi.WiDataFrame(df, **sar_d)
    assert (df == df2).all(axis=None)
    assert (df2.index == df.index).all()
    assert (df2.columns == df.columns).all()
    assert df2.sar_d == sar_d

    with pytest.raises(ValueError, match="rewards can have at most two columns"):
        wi.WiDataFrame(df2, states=df2.states, actions=df2.actions, rewards=["x", "y", "z"])


@pytest.mark.parametrize("copy_sar_d", (True, False))
def test_from_another_df(df, copy_sar_d):
    if not copy_sar_d:
        states = [s.upper() for s in df.sar_d["states"]]
        actions = [s.upper() for s in df.sar_d["actions"]]
        rewards = [s.upper() for s in df.sar_d["rewards"]]
        df2 = wi.WiDataFrame(df, states, actions, rewards)
        assert df2.sar_d == dict(states=states, actions=actions, rewards=rewards)
    else:
        df2 = wi.WiDataFrame(df)
        assert df2.sar_d == df.sar_d


def test_from_pd_series(data, sar_d):
    ser = pd.Series(data["a"], name="a")
    ser2 = wi.WiSeries(ser, **sar_d)
    assert (ser2 == ser).all()
    assert (ser2.index == ser.index).all
    assert ser2.name == ser.name
    assert ser2.sar_d == sar_d

    with pytest.raises(ValueError, match="rewards can have at most two columns"):
        wi.WiSeries(ser2, states=ser2.states, actions=ser2.actions, rewards=["x", "y", "z"])


@pytest.mark.parametrize("copy_sar_d", (True, False))
def test_from_another_series(df, copy_sar_d):
    ser = df["a"]
    if not copy_sar_d:
        states = [s.upper() for s in df.sar_d["states"]]
        actions = [s.upper() for s in df.sar_d["actions"]]
        rewards = [s.upper() for s in df.sar_d["rewards"]]
        ser2 = wi.WiSeries(ser, states, actions, rewards)
        assert ser2.sar_d == dict(states=states, actions=actions, rewards=rewards)
    else:
        ser2 = wi.WiSeries(ser)
        assert ser2.sar_d == ser.sar_d


def test_property_sar(df, sar):
    assert df.sar == sar


@pytest.mark.parametrize("sarsa", (True, False))
def test_add_value(df, sarsa, sar_d):
    expected_cols = df.columns.append(pd.Index(["value"]))

    df2 = df.copy()
    actual_cols = df2.add_value(sarsa=sarsa).columns
    assert np.all(actual_cols == expected_cols)
    assert df2.sar_d == {
        "states": sar_d["states"],
        "actions": sar_d["actions"],
        "rewards": sar_d["rewards"] + ["value"],
    }


def test_add_value_conflict(df):
    with pytest.raises(ValueError, match="Unspecified reward column:"):
        wi.WiDataFrame(df, states=df.states, actions=df.actions).add_value()

    with pytest.raises(ValueError, match="Unspecified reward column:"):
        wi.WiDataFrame(df, states=df.states, actions=df.actions, rewards=[]).add_value()

    df2 = df.copy()
    with pytest.raises(ValueError, match="conflicts with reward"):
        df2.add_value(value_col=df.rewards[0])

    # Test value columns are silently overriden
    cols_without_value = df.columns.tolist()
    for _ in range(2):
        df2.add_value(value_col="z")
        assert df2.columns.tolist() == cols_without_value + ["z"]

    with pytest.raises(ValueError, match="Unknown override: asdf"):
        df2.add_value(value_col="z", override="asdf")

    with pytest.raises(ValueError, match="Column z already exists"):
        df2.add_value(value_col="z", override="error")

    with pytest.warns(UserWarning, match="Column z will be overwritten"):
        df2.add_value(value_col="z", override="warn")


def test_add_value_alpha_gamma(df):
    for value in (-0.5, 1.5):
        with pytest.raises(ValueError, match="^Learning rate alpha="):
            df.add_value(alpha=value)

        with pytest.raises(ValueError, match="^Discount factor gamma="):
            df.add_value(gamma=value)

    for value in (0.0, 0.5, 1.0):
        df.add_value(alpha=value)
        df.add_value(gamma=value)


def test_df(df, sar_d, data):
    assert df.sar_d == sar_d
    assert df.states == sar_d["states"]
    assert df.actions == sar_d["actions"]
    assert df.rewards == sar_d["rewards"]
    assert df.values.shape == df.shape == (len(data["a"]), len(data))

    for k, v in data.items():
        ser = df[k]
        assert (ser == v).all()
        assert (ser.values == v).all()
        assert (
            isinstance(ser.values, np.ndarray)
            and ser.values.shape == (len(v),)
            and ser.values.dtype == int
        )
        assert ser.sar_d == sar_d
        assert ser.states == sar_d["states"]
        assert ser.actions == sar_d["actions"]
        assert ser.rewards == sar_d["rewards"]


def test_series(data, sar_d):
    for k, v in data.items():
        ser = wi.WiSeries(v, name=k, **sar_d)
        assert (ser == v).all()
        assert (ser.values == v).all()
        assert (
            isinstance(ser.values, np.ndarray)
            and ser.values.shape == (len(v),)
            and ser.values.dtype == int
        )
        assert ser.sar_d == sar_d
        assert ser.states == sar_d["states"]
        assert ser.actions == sar_d["actions"]
        assert ser.rewards == sar_d["rewards"]


def test_slice_df_to_series(df):
    for _, ser in df.items():
        assert isinstance(ser, wi.WiSeries)
        assert ser.sar == df.sar


def test_ser2df(data, sar_d):
    ser = wi.WiSeries(data["a"], **sar_d)
    assert ser.to_frame().sar_d == sar_d


def test_trim(df):
    df2 = df.trim()
    assert set(df2.columns) == set(df.sar)
    assert df2.shape == (df.shape[0], len(df.sar))


def test_groupby(df):
    g = df.groupby("a")

    gdf = g.median()
    # print("\n", gdf, sep="")
    assert g.median().shape == (3, 5)

    gdf = g.agg("sum")
    # print("\n", gdf, sep="")
    assert gdf.shape == (3, 5)

    gdf = g.agg(["min", "max"])
    # print("\n", gdf, sep="")
    assert gdf.shape == (3, 5 * 2)

    gdf = g.agg({"b": "sum"})
    # print("\n", gdf, sep="")
    assert gdf.shape == (3, 1)

    gdf = g.agg({"b": "sum", "c": "sum"})
    # print("\n", gdf, sep="")
    assert gdf.shape == (3, 2)


def test_concat_axis_1(df):
    df2 = pd.concat(
        [
            df[["a", "b"]],
            df[["c", "d"]],
            df["e"],
            df["f"],
        ],
        axis=1,
    )

    assert df2.shape == df.shape
    assert df2.sar_d == df.sar_d


def test_concat_axis_0(df):
    df2 = pd.concat([df, df], axis=0)
    assert df2.shape == (2 * df.shape[0], df.shape[1])
    assert df2.sar_d == df.sar_d


@pytest.mark.parametrize("dirname_as_str", (True, False))
def test_df_read_write_csv(df, tmpdir, dirname_as_str):
    p = str(tmpdir / "mydataset") if dirname_as_str else tmpdir / "mydataset"
    df.to_csv_dataset(p, index=False)
    df2 = wi.read_csv_dataset(p)
    assert (df == df2).all(axis=None)
    assert df.sar_d == df2.sar_d
    assert df2.apply(is_numeric_dtype).all()

    # With forced_categories
    p = str(tmpdir / "mydataset") if dirname_as_str else tmpdir / "mydataset-all-cat"
    df.to_csv_dataset(p, forced_categories=df.columns, index=False)
    df2 = wi.read_csv_dataset(p)
    assert (df.astype(str) == df2).all(axis=None)
    assert df.sar_d == df2.sar_d
    assert df2.apply(is_object_dtype).all()


@pytest.mark.parametrize("dirname_as_str", (True, False))
def test_ser_read_write_csv(df, tmpdir, dirname_as_str):
    p = str(tmpdir / "mydataset") if dirname_as_str else tmpdir / "mydataset"
    df.iloc[:, 0].to_csv_dataset(p, index=False, header=True)
    df2 = wi.read_csv_dataset(p)
    assert df2.shape[1] == 1
    ser = df2.iloc[:, 0]
    assert (df.iloc[:, 0] == ser).all()
    assert df.sar_d == ser.sar_d
    assert is_numeric_dtype(ser.dtype)

    # With forced_categories
    p = str(tmpdir / "mydataset") if dirname_as_str else tmpdir / "mydataset-all-cat"
    df.iloc[:, 0].to_csv_dataset(p, forced_categories=df.columns, index=False)
    df2 = wi.read_csv_dataset(p)
    assert df2.shape[1] == 1
    ser = df2.iloc[:, 0]
    assert (df.iloc[:, 0].astype(str) == ser).all(axis=None)
    assert df.sar_d == ser.sar_d
    assert is_object_dtype(ser.dtype)


def test_metadata_obj(df, tmpdir):
    m = wi.Metadata(**df.sar_d)
    wi.save_metadata(m, tmpdir / "metadata.yaml")
    m2 = wi.read_metadata(tmpdir / "metadata.yaml")
    assert m == m2


def test_overwrite_datadir(df, tmpdir):
    with pytest.warns(UserWarning, match=r"Existing directory .* will be overwritten"):
        df.to_csv_dataset(tmpdir, index=False)


def test_metadata_dict_correct(df, tmpdir):
    # A few different syntax for typing.TypedDict. Most doesn't work with current mypy.
    # m: wi.MetadataDict = {**df.sar_d, "forced_categories": None, "frequency": None, "tags": {}}
    # m = wi.MetadataDict(**df.sar_d, forced_categories=None, frequency=None, tags={})
    m = wi.MetadataDict(
        states=df.states,
        actions=df.actions,
        rewards=df.rewards,
        forced_categories=None,
        frequency=None,
        tags={},
    )

    wi.save_metadata(m, tmpdir / "metadata.yaml")
    m2 = wi.read_metadata(tmpdir / "metadata.yaml")
    assert m2 == wi.Metadata(**df.sar_d)

    wi.save_metadata(m, tmpdir / "metadata-compact.yaml", compact=True)
    m3 = wi.read_metadata(tmpdir / "metadata-compact.yaml")
    assert m2 == m3

    with open(tmpdir / "metadata.yaml") as f:
        d2 = yaml.safe_load(f)

    with open(tmpdir / "metadata-compact.yaml") as f:
        d3 = yaml.safe_load(f)

    assert d2 == m
    assert d3 == {**df.sar_d, "tags": {}}


def test_metadata_dict_wrong(df, tmpdir):
    with pytest.raises(TypeError):
        wi.save_metadata(df.sar_d, tmpdir / "metadata.yaml")


def test_sequence(df):
    seq = df.sequence
    assert seq.shape == (df.shape[0] * df.shape[1],)
    assert (df.values.ravel() == seq).all()


def test_plot(df):
    df.plot()
