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

from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from typeguard import check_type

import a2rl as wi

from ._metadatadict import MetadataDict


def sample_dataset_path(dataset_name: str) -> Path:
    """Resolve the path to the sample dataset.

    Args:
        dataset_name: Name of sample dataset.

    Returns:
        Path to the directory of ``dataset_name``.

    See Also
    --------
    list_sample_datasets
    read_csv_dataset


    Examples:

        .. code-block:: python

            >>> import pandas as pd
            >>> import a2rl as wi
            >>> p = wi.sample_dataset_path('chiller')
            >>> p
            PosixPath('.../a2rl/dataset/chiller')

            >>> df = wi.read_csv_dataset(p).trim()
            >>> with pd.option_context('display.max_columns', 2):
            ...     print(df.head())
               condenser_inlet_temp  ...  system_power_consumption
            0                  29.5  ...                     756.4
            1                  30.2  ...                     959.3
            2                  29.3  ...                     586.1
            3                  28.5  ...                    1178.5
            4                  30.3  ...                     880.9
            <BLANKLINE>
            [5 rows x 4 columns]

            >>> df.shape
            (9153, 4)
    """
    return Path(__file__).parent / "dataset" / dataset_name


def list_sample_datasets() -> list[str]:
    """List the name of sample datasets included in ``whatif``.

    Returns:
        Dataset names

    See Also
    --------
    sample_dataset_path


    Examples:

        .. code-block:: python

            >>> import a2rl as wi
            >>> wi.list_sample_datasets()
            ['chiller', 'rtu']
    """
    prefix = Path(__file__).parent / "dataset"
    file_list = [metadata.parent.name for metadata in prefix.glob("*/metadata.yaml")]
    file_list.sort()
    return file_list


@dataclass
class Metadata:
    """Metadata of a ``Whatif`` dataframe or dataset.

    Arguments:
        states: Column names for states.
        actions: Column names for actions.
        rewards: Column names for rewards.
        forced_categories: Numeric columns that must be interpreted as categorical or ordinal.
            Otherwise, column dtypes are automatically determines.
        frequency: Sampling frequency of the dataset, in the Pandas frequency string format. See the
            ``freq`` argument in
            :func:`pandas.tseries.frequencies.to_offset`, and the :pdug:`pandas DateOffset tutorial
            <timeseries.html#dateoffset-objects>`. Examples: ``H``, ``2H``, ``D``.
        tags: Additional custom metadata. Defaults to an empty dictionary ``{}``.

    See Also
    --------
    read_metadata
    save_metadata


    Examples:

        Create an in-memory metadata object.

            .. code-block:: python

                >>> import a2rl as wi
                >>> m = wi.Metadata(
                ...     states=["s", "t"],
                ...     actions=["a"],
                ...     rewards=["r"],
                ...     frequency="H",
                ... )
                >>> m  # doctest: +NORMALIZE_WHITESPACE
                Metadata(states=['s', 't'], actions=['a'], rewards=['r'], forced_categories=None,
                frequency='H', tags={})

        Create from a dictionary with default sampling frequency and tags.

            .. code-block:: python

                >>> d = {
                ...     "states": ["s", "t"],
                ...     "actions": ["a"],
                ...     "rewards": ["r"],
                ... }
                >>> wi.Metadata(**d)  # doctest: +NORMALIZE_WHITESPACE
                Metadata(states=['s', 't'], actions=['a'], rewards=['r'], forced_categories=None,
                frequency=None, tags={})

        Convert the metadata object to a YAML string. Please note this is shown for pedagogical
        purpose only. In practice, we recommend :func:`read_metadata` and :func:`save_metadata`
        to convert between :class:`Metadata` and YAML file.

            .. code-block:: python

                >>> from dataclasses import asdict
                >>> import yaml
                >>> s = yaml.safe_dump(asdict(m), sort_keys=False)
                >>> print(s)
                states:
                - s
                - t
                actions:
                - a
                rewards:
                - r
                forced_categories: null
                frequency: H
                tags: {}
                <BLANKLINE>
    """

    states: list[str]  #: ``list[str]`` - Column names for states.
    actions: list[str]  #: ``list[str]`` - Column names for actions.
    rewards: list[str]  #: ``list[str]`` - Column names for rewards.

    #: ``None | list[str]`` -- Numerical column names that must be interpreted as categorical or
    #: ordinal.
    forced_categories: None | list[str] = None

    #: ``None | str`` - Sampling frequency of the dataset, in the Pandas frequency string format.
    #: See the ``freq`` argument in :func:`pandas.tseries.frequencies.to_offset`, and the
    #: :pdug:`pandas DateOffset tutorial <timeseries.html#dateoffset-objects>`. Examples: ``H``,
    #: ``2H``, ``D``.
    frequency: None | str = None

    #: ``dict[str, Any]`` - Additional custom metadata.
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_type("states", self.states, List[str])
        check_type("actions", self.actions, List[str])
        check_type("rewards", self.rewards, List[str])
        check_type("forced_categories", self.forced_categories, Optional[List[str]])
        check_type("frequency", self.frequency, Optional[str])
        check_type("tags", self.tags, Dict[str, Any])


def read_metadata(yaml_file: str | Path) -> Metadata:
    """Load a YAML file into an in-memory metadata object.

    Arguments:
        yaml_file: Path to the input YAML file.

    Returns:
        In-memory, metadata object

    See Also
    --------
    Metadata
    save_metadata


    Examples:

        Read the metadata of the ``chiller`` sample dataset.

        .. code-block:: python

            >>> import a2rl as wi
            >>> p = wi.sample_dataset_path("chiller") / "metadata.yaml"
            >>> m = wi.read_metadata(p)
            >>> m.states
            ['condenser_inlet_temp', 'evaporator_heat_load_rt']
    """
    p = yaml_file if isinstance(yaml_file, Path) else Path(yaml_file)
    with p.open("r") as f:
        d = yaml.safe_load(f)
    return Metadata(**d)


def save_metadata(
    metadata: Metadata | MetadataDict,
    yaml_file: str | Path,
    compact: bool = False,
) -> None:
    """Save an in-memory metadata object into a YAML file.

    Arguments:
        metadata: Metadata object.
        yaml_file: Path to the output YAML file.
        compact: When set to True, do not output ``None`` entries.

    See Also
    --------
    Metadata
    read_metadata


    Examples:

        Save an in-memory metadata object.

        .. code-block:: python

            >>> import a2rl as wi
            >>> m = wi.Metadata(
            ...     states=["s", "t"],
            ...     actions=["a"],
            ...     rewards=["r"],
            ...     tags={"k": "v"}
            ... )
            >>> wi.save_metadata(m, "/tmp/metadata.yaml")

            >>> with open("/tmp/metadata.yaml") as f:
            ...     print(''.join(f.readlines()))
            states:
            - s
            - t
            <BLANKLINE>
            actions:
            - a
            <BLANKLINE>
            rewards:
            - r
            <BLANKLINE>
            forced_categories: null
            <BLANKLINE>
            frequency: null
            <BLANKLINE>
            tags:
              k: v
            <BLANKLINE>


        Save metadata in compact mode to exclude null items in the YAML output.

        .. code-block:: python

            >>> wi.save_metadata(m, "/tmp/metadata.yaml", compact=True)
            >>> with open("/tmp/metadata.yaml") as f:
            ...     print(''.join(f.readlines()))
            states:
            - s
            - t
            <BLANKLINE>
            actions:
            - a
            <BLANKLINE>
            rewards:
            - r
            <BLANKLINE>
            tags:
              k: v
            <BLANKLINE>

        Save a dictionary. Be aware that the dictionary **must** specifies **all** the
        :class:`Metadata` fields, including the default ones. If you prefer the flexibility to not
        re-declare the default fields, please use :class:`Metadata` instead.

        .. code-block:: python

            >>> d: wi.MetadataDict = {
            ...     "states": ["s", "t"],
            ...     "actions": ["a"],
            ...     "rewards": ["r"],
            ...     "forced_categories": ["a"],
            ...     "frequency": None,
            ...     "tags": {},
            ... }
            >>> wi.save_metadata(d, "/tmp/metadata.yaml", compact=True)

            >>> with open("/tmp/metadata.yaml") as f:
            ...     print(''.join(f.readlines()))
            states:
            - s
            - t
            <BLANKLINE>
            actions:
            - a
            <BLANKLINE>
            rewards:
            - r
            <BLANKLINE>
            forced_categories:
            - a
            <BLANKLINE>
            tags: {}
            <BLANKLINE>
    """
    # Based on https://github.com/yaml/pyyaml/issues/127#issuecomment-525800484
    class BlankLiner(yaml.SafeDumper):
        def write_line_break(self, data=None):
            super().write_line_break(data)

            if len(self.indents) == 1:
                super().write_line_break()

    if isinstance(metadata, Metadata):
        m = metadata
    elif isinstance(metadata, dict):
        check_type("metadata_dictionary", metadata, MetadataDict)
        m = Metadata(**metadata)

    p = yaml_file if isinstance(yaml_file, Path) else Path(yaml_file)
    d = asdict(m)
    if compact:
        d = {k: v for k, v in d.items() if v is not None}
    with p.open("w") as f:
        yaml.dump(d, f, sort_keys=False, Dumper=BlankLiner)


def read_csv_dataset(
    dirpath: str | PathLike[str],
    *args,
    test_mdp: bool = False,
    low_memory: bool = False,
    **kwargs,
) -> wi.WiDataFrame:
    """Read a dataset directory into a :class:`a2rl.WiDataFrame`.

    Args:
        dirpath: Path to the dataset directory.
        *args: Positional arguments passed as-is to :func:`pandas.read_csv`.
        mdp_test: When ``True``, perform Markovian self-check on the dataframe loaded.
            Raise :exc:`a2rl.utils.NotMDPDataError` if the check fails.
        low_memory: If ``False``, read the entire .csv payload. If ``True``, internally process the
            .csv payload in chunks. This argument is passed as-is to :func:`pandas.read_csv`,
            however note that we override the default to ``True``, which is opposite to the default
            in :func:`pandas.read_csv`.
        **kwargs: Keyword arguments passed as-is to :func:`pandas.read_csv`.

    Returns:
        The loaded dataset.

    See Also
    --------
    sample_dataset_path
    WiDataFrame.to_csv_dataset


    Examples:

        .. code-block:: python

            >>> import a2rl as wi
            >>> p = wi.sample_dataset_path('chiller')
            >>> df = wi.read_csv_dataset(p)
            >>> df.info()  # doctest: +NORMALIZE_WHITESPACE
            <class 'a2rl._dataframe.WiDataFrame'>
            RangeIndex: 9153 entries, 0 to 9152
            Data columns (total 5 columns):
             #   Column                    Non-Null Count  Dtype
            ---  ------                    --------------  -----
             0   timestamp                 9153 non-null   object
             1   staging                   9153 non-null   object
             2   condenser_inlet_temp      9153 non-null   float64
             3   evaporator_heat_load_rt   9153 non-null   float64
             4   system_power_consumption  9153 non-null   float64
            dtypes: float64(3), object(2)
            memory usage: ... KB
    """
    p = dirpath if isinstance(dirpath, Path) else Path(dirpath)
    metadata = read_metadata(p / "metadata.yaml")
    files = p.glob("**/*.csv")

    kwargs["low_memory"] = low_memory
    if metadata.forced_categories:
        kwargs["dtype"] = {col: str for col in metadata.forced_categories}
    dfs = {str(fpath): pd.read_csv(fpath, *args, **kwargs) for fpath in files}

    df = wi.WiDataFrame(
        pd.concat(dfs.values()),
        states=metadata.states,
        actions=metadata.actions,
        rewards=metadata.rewards,
    )

    if test_mdp:
        tokeniser = wi.DiscreteTokenizer(n_bins=50)
        df_tok = tokeniser.fit_transform(df.trim())
        wi.utils.assert_mdp(df_tok)

    return df
