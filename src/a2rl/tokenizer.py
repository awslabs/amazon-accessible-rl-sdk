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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.utils import check_array

import a2rl as wi


class Tokenizer(ABC):
    """Abstract base class of tokenizers."""

    @abstractmethod
    def fit(self, df: wi.WiDataFrame) -> Tokenizer:
        """Fit this tokenizer.

        Args:
            df: Training data.

        Returns:
            This fitted tokenizer.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: wi.WiDataFrame) -> wi.WiDataFrame:
        """Tokenize a data frame.

        Args:
            df: Data-frame to tokenized.

        Returns:
            Tokenized data frame.
        """
        raise NotImplementedError

    def fit_transform(self, df: wi.WiDataFrame) -> wi.WiDataFrame:
        """Call :meth:`fit()` then :meth:`transform()`.

        Args:
            df: Data-frame used as the training data for fitting this tokenizer, and then to be
                tokenized.

        Returns:
            Tokenized data frame.
        """
        self.fit(df)
        return self.transform(df)

    @abstractmethod
    def inverse_transform(self, df: wi.WiDataFrame) -> wi.WiDataFrame:
        """Revert the tokenized data frame back to their original space.

        Args:
            df: Tokenized data frame.

        Returns:
            Data frame in the original space.
        """
        raise NotImplementedError

    def valid_tokens(self, col: str | int) -> list[int | np.integer]:
        """Get the valid tokens for column ``col``.

        Arguments:
            col: Column name (str) or column index (int).

        Returns:
            Valid tokens for column ``col``.
        """
        if isinstance(col, int):
            return self.valid_tokens_of_col_idx(col)
        else:
            return self.valid_tokens_of_col_name(col)

    @abstractmethod
    def valid_tokens_of_col_idx(self, col_idx: int) -> list[int | np.integer]:
        """Get the valid tokens for column index ``col_idx``.

        Arguments:
            col_idx: column index.

        Returns:
            Valid tokens for column index ``col_idx``.
        """
        raise NotImplementedError

    @abstractmethod
    def valid_tokens_of_col_name(self, col_name: str) -> list[int | np.integer]:
        """Get the valid tokens for column name ``col_name``.

        Arguments:
            col_name: column name.

        Returns:
            Valid tokens for column name ``col_name``.
        """
        raise NotImplementedError


def compute_bases(bins_per_column: np.ndarray) -> np.ndarray:
    """Compute the base of each column.

    The base is added to each column to transform local bucket indexes to globally-unique tokens.

    Consider this example of a dataframe that has been discretized:

    .. code-block:: text

        - Column 0 has 3 buckets, and the bucket identifiers are in [0, 1, 2]
        - Column 1 has 2 buckets, and the bucket identifiers are in [0, 1]
        - Column 2 has 4 buckets, and the bucket identifiers are in [0, 1, 2, 3]

    We want each column to have unique bucket identifiers by using the bases:

    .. code-block:: text

        - Column 0 has global bucket identifiers [0, 1, 2]    + 0 = [0, 1, 2]
        - Column 1 has global bucket identifiers [0, 1]       + 3 = [3, 4]
        - Column 2 has global bucket identifiers [0, 1, 2, 3] + 4 = [5, 6, 7, 8]

    Args:
        bins_per_column: The number of discrete bins of each column.

    Raises:
        ValueError: If ``bins_per_column`` is not a 1D array.

    Returns:
        Base bins of each columns.
    """
    if len(bins_per_column.shape) != 1:
        raise ValueError(
            f"Expect 1D array, but getting bins_per_column whose shape={bins_per_column.shape}"
        )

    bases = np.zeros(len(bins_per_column), dtype=bins_per_column.dtype)
    bases[1:] = bins_per_column[0:-1]
    bases = bases.cumsum()
    return bases


@dataclass
class DiscreteTokenizer(Tokenizer):
    """Discretize numeric columns and label encode categorical columns.

    The resulted tokens are unique across columns.

    By default, the fitting step asserts every column to have sufficient variance (i.e., more than
    one unique value). Although this check can be switched off, you're still strongly recommended
    against it, as single-value columns brings no information gain to the optimization process.
    As an example, there's nothing offline RL can learn from a historical data with a constant
    action. The same arguments applies for states and rewards as well.

    Arguments:
        n_bins: number of discrete bins for continuous columns.
        num_bins_strategy: Discretization strategy as in
            :class:`sklearn.preprocessing.KBinsDiscretizer`

    Examples:

        .. code-block:: python

            >>> from a2rl import WiDataFrame, DiscreteTokenizer
            >>> df = WiDataFrame(
            ...     data=[
            ...         [ 10,  5,  20, "x", "a"],
            ...         [ 10,  5,  40, "x", "a"],
            ...         [ 50,  5,  50, "y", "b"],
            ...         [ 50, 85,  60, "y", "b"],
            ...         [ 90, 85,  80, "z", "b"],
            ...         [ 90, 85, 100, "z", "a"],
            ...     ],
            ...     columns=list("ABCDE"),
            ...     states=["s1", "s2"],
            ...     actions=["a"],
            ...     rewards=["r"],
            ... )
            >>> df
                A   B    C  D  E
            0  10   5   20  x  a
            1  10   5   40  x  a
            2  50   5   50  y  b
            3  50  85   60  y  b
            4  90  85   80  z  b
            5  90  85  100  z  a

            >>> t = DiscreteTokenizer(n_bins=5, num_bins_strategy="uniform").fit(df)
            >>> df_tok = t.transform(df)
            >>> df_tok
               A  B   C   D   E
            0  0  5  10  15  18
            1  0  5  11  15  18
            2  2  5  11  16  19
            3  2  9  12  16  19
            4  4  9  13  17  19
            5  4  9  14  17  18

        Fit-transform in one go.

        .. code-block:: python

            >>> t.fit_transform(df)
               A  B   C   D   E
            0  0  5  10  15  18
            1  0  5  11  15  18
            2  2  5  11  16  19
            3  2  9  12  16  19
            4  4  9  13  17  19
            5  4  9  14  17  18

        Reconstruct the approximated original data frame.

        .. code-block:: python

            >>> t.inverse_transform(df_tok)
                  A     B     C  D  E
            0  18.0  13.0  28.0  x  a
            1  18.0  13.0  44.0  x  a
            2  50.0  13.0  44.0  y  b
            3  50.0  77.0  60.0  y  b
            4  82.0  77.0  76.0  z  b
            5  82.0  77.0  92.0  z  a
    """

    #: ``int`` - Number of discrete bins for continuous columns.
    n_bins: int = 100

    #: ``str`` - Discretization strategy as per :class:`sklearn.preprocessing.KBinsDiscretizer`.
    num_bins_strategy: str = "quantile"

    #: ``list[str]`` - Columns recognized by this tokenizer.
    columns: list[str] = field(init=False, default_factory=list)

    #: ``list[str]`` - Categorical columns recognized by this tokenizer.
    cat_columns: list[str] = field(init=False, default_factory=list)

    #: ``list[str]`` - Numerical columns recognized by this tokenizer.
    quantized_columns: list[str] = field(init=False, default_factory=list)

    #: ``np.ndarray`` - Base bins for numberical columns.
    bases_num: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=int))

    #: ``np.ndarray`` - Base bins for categorical columns.
    bases_cat: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=int))

    _label_encoder: OrdinalEncoder = field(init=False, repr=False)
    _quantizer: KBinsDiscretizer = field(init=False, repr=False)

    def check_numerical_columns(self, df: pd.DataFrame) -> None:
        """Input validation on the all-numerical input dataframe.

        Each column in the input dataframe must contain only finite values, and it cannot have just
        a single unique value. Callers are responsible to ensure the input dataframe contains only
        numeric columns.

        These are considered non-finite values: `None`, :class:`numpy.nan`, :class:`numpy.inf`,
        :class:`pandas.NA` (i.e., the nullable integers).

        Args:
            df: an input dataframe whose all columns must be numeric. Callers must ensure to pass
                an all-numerical input dataframe.
        """
        violations = []
        for c, ser in df.items():
            try:
                # Re-implement exception in KBinsDiscretizer.fit() for friendlier error message.
                # This sklearn ensures only valid numbers are present (i.e., no +/- inf, nan).
                check_array(df)
            except (
                ValueError,
                TypeError,  # Caused by pandas.NA.
            ) as e:
                raise ValueError(
                    f"One or more numerical columns in {df.columns.tolist()} has problems: {e}"
                )

            # See: KBinsDiscretizer.fit() in sklearn/preprocessing/_discretization.py.
            if ser.min() == ser.max():
                # Single value causes inverse_transform() to produce nan.
                violations.append(c)

        if len(violations) > 0:
            raise ValueError(f"Single numerical values detected on columns {violations}")

    def check_categorical_columns(self, df: pd.DataFrame) -> None:
        """Input validation on the all-categorical input dataframe.

        Each column in the input dataframe must have more than one unique values (which may include
        ``None`` and ``pandas NA``). Callers are responsible to ensure the input dataframe contains
        only non-numeric columns.

        Args:
            df: an input dataframe whose all columns must be non-numeric. Callers must ensure to
                pass an all-non-numerical input dataframe.
        """
        violations = [c for c, ser in df.items() if ser.nunique(dropna=False) < 2]
        if len(violations) > 0:
            raise ValueError(f"Single categorical values detected on columns {violations}")

    def fit(self, df: wi.WiDataFrame, check: bool = True) -> DiscreteTokenizer:
        """Fit the quantizer for the numeric columns, and the label encoder for the categorical
        columns.

        Args:
            df: Training data.
            check: When ``True``, ensure that ``df`` contains sufficient variance (i.e., a column
                must not have just a single value), and numerical columns contains only finite
                values.

        Returns:
            This fitted discrete tokenizer.

        Raises:
            ValueError: when ``check=True`` and violations found on input data.

        See Also
        --------
        check_numerical_columns : Checks performed on numerical columns.
        check_categorical_columns : Checks performed on categorical columns.


        Examples
        --------
        Fitting a dataframe with enough variance (i.e., more than one unique values).

        .. code-block:: python

            >>> import a2rl as wi
            >>> from a2rl.utils import tokenize
            >>>
            >>> wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller")).trim()
            >>> wi_df.nunique()  # doctest: +NORMALIZE_WHITESPACE
            <BLANKLINE>
            condenser_inlet_temp          70
            evaporator_heat_load_rt     5279
            staging                       11
            system_power_consumption    5354
            dtype: int64

            >>> tok = wi.DiscreteTokenizer().fit(wi_df)

        An example of fitting a dataframe with not enough variance. In this example, the training
        data has just one single action.

        .. code-block:: python

            >>> df_constant_action = wi_df.head().copy()
            >>> df_constant_action["staging"] = "0"
            >>> df_constant_action.nunique()  # doctest: +NORMALIZE_WHITESPACE
            <BLANKLINE>
            condenser_inlet_temp        5
            evaporator_heat_load_rt     5
            staging                     1
            system_power_consumption    5
            dtype: int64

            >>> wi.DiscreteTokenizer().fit(df_constant_action)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Single numerical values detected on columns ['staging']
        """
        # Track the columns
        self.columns = df.columns.tolist()
        self.cat_columns = [c for c in df.columns if not is_numeric_dtype(df[c])]
        cat_columns = set(self.cat_columns)
        self.quantized_columns = [c for c in self.columns if c not in cat_columns]

        # Fit the quantizer and the disambiguation constants.
        min_cat_token = 0
        if len(self.quantized_columns) > 0:
            df_num = df[self.quantized_columns]
            if check:
                self.check_numerical_columns(df_num)

            # Fit quantizer with array. The quantizer will lose column names, however, it prevents
            # warnings when a df-fitted estimator has its transform() called with an np.ndarray.
            self._quantizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy=self.num_bins_strategy,
            ).fit(df_num.values)

            # Compute disambiguation constants
            self.bases_num = compute_bases(self._quantizer.n_bins_).reshape((1, -1))
            min_cat_token = self.bases_num[0, -1] + self._quantizer.n_bins_[-1]

        # Fit the label encoder and the disambiguation constants.
        if len(self.cat_columns) > 0:
            df_cat = df[self.cat_columns]
            if check:
                self.check_categorical_columns(df_cat)
            self._label_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int64,
            )
            self._label_encoder.fit(df_cat.values)

            # Compute disambiguation constants
            cat_count = [len(arr) for arr in self._label_encoder.categories_]
            bases_cat = compute_bases(np.asarray(cat_count)) + min_cat_token
            self.bases_cat = bases_cat.reshape((1, -1))

        return self

    def transform(self, df: wi.WiDataFrame) -> wi.WiDataFrame:
        """Discretize the numberic columns into tokens, label encode the categorical columns into
        tokens, then disambiguate the tokens across all columns.

        Args:
            df: Data-frame to discretized.

        Returns:
            Tokenized data frame.
        """
        sar_d = df.sar_d
        df_encoded = df_quantized = wi.WiDataFrame(**sar_d)

        # Quantize the numerical columns.
        if len(self.cat_columns) > 0:
            df_encoded = wi.WiDataFrame(
                self._label_encoder.transform(df[self.cat_columns].values),
                columns=self.cat_columns,
                **sar_d,
            )
            df_encoded += self.bases_cat

        # Label-encode the categorical columns.
        if len(self.quantized_columns) > 0:
            df_quantized = wi.WiDataFrame(
                self._quantizer.transform(df[self.quantized_columns].values).astype(int),
                columns=self.quantized_columns,
                **sar_d,
            )
            df_quantized += self.bases_num

        return pd.concat([df_quantized, df_encoded], axis=1)[self.columns]

    def fit_transform(self, df: wi.WiDataFrame, check: bool = True) -> wi.WiDataFrame:
        """Call :meth:`fit()` then :meth:`transform()`.

        Args:
            df: Data-frame used as the training data for fitting this tokenizer, and then to be
                tokenized.
            check: When ``True``, ensure that ``df``, when used for fitting, contains sufficient
                variance (i.e., a column must not have just a single value), and numerical columns
                contains only finite values.

        Returns:
            Tokenized data frame.

        Raises:
            ValueError: when ``check=True`` and violations found on input data.

        See Also
        --------
        check_numerical_columns : Checks performed on numerical columns.
        check_categorical_columns : Checks performed on categorical columns.
        """
        self.fit(df, check)
        return self.transform(df)

    def inverse_transform(self, df: wi.WiDataFrame) -> wi.WiDataFrame:
        """Revert the tokenized (i.e., discretized) data-frame bins back to their original space.

        Due to discretization, the reconstructed numerical columns may not match to the original
        undiscretized data frame.

        Args:
            df: Tokenized data frame.

        Returns:
            Data frame in the original space (approximation).
        """
        sar_d = df.sar_d
        df_num = df_cat = wi.WiDataFrame(**sar_d)

        # Reconstruct numerical columns
        if len(self.quantized_columns) > 0:
            df_num = df[self.quantized_columns] - self.bases_num
            df_num = wi.WiDataFrame(
                self._quantizer.inverse_transform(df_num.values),
                columns=self.quantized_columns,
                **sar_d,
            )

        # Reconstruct categorical columns
        if len(self.cat_columns) > 0:
            df_cat = df[self.cat_columns] - self.bases_cat
            df_cat = wi.WiDataFrame(
                self._label_encoder.inverse_transform(df_cat.values),
                columns=self.cat_columns,
                **sar_d,
            )

        return pd.concat([df_num, df_cat], axis=1)[self.columns]

    def valid_tokens_of_col_idx(self, col_idx: int) -> list[int | np.integer]:
        col_name = self.columns[col_idx]
        return self.valid_tokens_of_col_name(col_name)

    def valid_tokens_of_col_name(self, col_name: str) -> list[int | np.integer]:
        if col_name in self.cat_columns:
            c = self.cat_columns.index(col_name)
            cat_count = len(self._label_encoder.categories_[c])
            return [self.bases_cat[0, c] + oc for oc in range(cat_count)]
        else:
            c = self.quantized_columns.index(col_name)
            return [
                self.bases_num[0, c] + discrete_bucket
                for discrete_bucket in range(self._quantizer.n_bins_[c])
            ]
