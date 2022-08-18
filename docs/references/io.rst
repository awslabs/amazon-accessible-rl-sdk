============
Input/Output
============

.. currentmodule:: a2rl

Dataset
-------

A *dataset* is a unit that consists of *data payload* and *metadata*.

Presently, ``whatif`` stores a dataset as a directory layout:

.. code-block::

    dataset_name/
    |-- metadata.yaml
    `-- data.csv

and loads the dataset into a :class:`a2rl.WiDataFrame`.

In future, ``whatif`` may add additional implementations to support different storage layout.

.. autosummary::
    :toctree: api
    :nosignatures:

    list_sample_datasets
    read_csv_dataset
    sample_dataset_path

Metadata
--------

The metadata API is primarily intended for ``whatif`` developers.

.. autosummary::
    :toctree: api
    :nosignatures:

    Metadata
    read_metadata
    save_metadata


.. container:: hidden

    .. autosummary::
        :toctree: api
        :nosignatures:

        _metadatadict.MetadataDict
