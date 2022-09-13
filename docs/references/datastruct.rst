===============
Data Structures
===============

.. currentmodule:: a2rl

``whatif`` extends ``pandas`` data frame and series to allow you enrich your tabular and sequence
data with *sar* information (i.e., the expected column names of*states*, *actions*, and *rewards*).

Please note that this site documents only functionalities that ``whatif`` specifically provides.
Should you need the documentations for the base ``pandas`` functionalities, please consult to
:class:`pandas.DataFrame` and :class:`pandas.Series`.

.. autosummary::
    :toctree: api
    :nosignatures:

    WiDataFrame
    WiSeries
    TransitionRecorder
