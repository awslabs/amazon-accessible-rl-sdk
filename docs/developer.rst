Developer Guide
===============

We welcome and encourage all kind of help, such as bug reports, bug fixes, documentation
improvements, enhancements, ideas, marketing and communications, devops, project coordination, etc.

Setup your work environment as follows:

.. code-block:: bash

    git clone https://github.com/awslabs/amazon-accessible-rl-sdk.git
    # Alternative: git clone git@github.com:awslabs/amazon-accessible-rl-sdk.git

    cd amazon-accessible-rl-sdk

    # Perform editable install (preferrably to a virtual environment) so that
    # changes to the code base will be reflected immediately.
    pip install -e .

Once the command completes, you'll have a local repository at ``./amazon-accessible-rl-sdk/``.
Apart from source codes, you should notice these additional requirement files:

==========================  =============================================
``requirements-test.txt``   Required for unit tests
``requirements-docs.txt``   Required to generate the html documentations.
``requirements-build.txt``  Required to build the `wheel <https://packaging.python.org/en/latest/glossary/#term-wheel>`_.
``requirements-dev.txt``    Recommended packages during development.
==========================  =============================================

Hygiene Practice
----------------

Although the project repository implements code checks in its CI process, you're still strongly
recommended to run the checks locally to quickly catch and fix low-hanging-fruit violations.

Required packages and tools can be installed as:

.. code-block:: bash

    pip install -r <(cat requirements-*.txt)

Linters
~~~~~~~

The code-base comes with a set of git `pre-commit <https://pre-commit.com/>`_ hooks listed in the
``.pre-commit-config.yaml`` file.

.. code-block:: bash

    pre-commit install

Unit Tests
~~~~~~~~~~

Unit tests are done using the `pytest <https://docs.pytest.org/en/stable/>`_ framework.

We require that your contributions do not break existing tests. And if you found the existing tests
are buggy, please report it or feel free to contribute your fixes.

You should also include new tests whenever it makes sense (and you'd be the one who makes the
judgement call).

As a rule of thumb:

- new tests for every new capability (e.g., new API, changing API)
- new tests for every bug reported as an issue, because this means that the existing tests were not
  exhaustive enough to catch the bug.

Below are a few typical ways to run ``pytest``. Please consult ``pytest`` documentation for more
details.

.. code-block:: bash

    # Run all unit tests
    pytest

    # Run all unit tests with increased verbosity
    pytest -v -rA

    # Run a specific test file
    pytest test/test_dataframe.py

    # Run a specific unit test
    pytest test/test_dataframe.py::test_df

    # Only run tests which match the given substring test names.
    pytest -k test_from

    # Similar to above, but only for unit tests in the given test file.
    pytest test/test_dataframe.py -k test_from

Code Coverage
~~~~~~~~~~~~~

Code coverage are computed using the `coverage <https://pytest-cov.readthedocs.io/en/stable/>`_
tool.

.. code-block:: bash

    coverage run -m pytest

    # View the summary
    coverage report

    # Generage a detailed html report to htmlcov/.
    # To view the detailed report, open htmlcov/index.html.
    coverage html

Type Checks
~~~~~~~~~~~

.. note::

    If you're new to type checks, we encourage you to learn more about its benefits and how-to on
    `mypy <https://mypy.readthedocs.io/en/stable/>`_ and
    `Python official documentation <https://peps.python.org/pep-0484/>`_.

As a pre-requisite, you need to enable type hints on your `pandas <https://pandas.pydata.org>`_
installations:

.. code-block:: bash

    # See: https://pandas.pydata.org/docs/dev/development/contributing_codebase.html#testing-type-hints-in-code-using-pandas
    python -c "import pandas; import pathlib; (pathlib.Path(pandas.__path__[0]) / 'py.typed').touch()"

Then, run the type checks as follows:

.. code-block:: bash

    mypy --install-types --config-file tox.ini --exclude '^.venv/.*' --exclude '^build/.*' .


HTML Docs
---------

Portion of the documentations are written as Jupyter notebooks (i.e., ``notebooks/*.ipynb`` files).
As such, you need to install `pandoc <https://pandoc.org>`_ on your computer by consulting their
`installation instructions <https://pandoc.org/installing.html>`_.

You also need `mermaid-cli <https://github.com/mermaid-js/mermaid-cli>`_ to build `Mermaid
<https://mermaid-js.github.io/>`_ diagrams. To install on your computer, please refer to its
installation instructions. **NOTE**: this is a workaround until this `issue
<https://github.com/mgaitan/sphinxcontrib-mermaid/issues/74>`_ is fixed.

We recommend that you check the correctness of the inline code in the API docstrings:

.. code-block:: bash

    pytest src/a2rl

Should you encounter failed cases, we highly encourage you to report this as a new issue.

Then, generate the html documentations as follows:

.. code-block:: bash

    cd docs
    make clean
    make html

On a multi-core machine, you can also pass the ``-j <num>`` to speed-up the build process.

.. code-block:: bash

    cd docs
    make clean

Once completed, you can view the generated html pages at ``docs/_build/html/index.html``.

.. note::

    Here're a few tricks to speed-up the build time, especially when writing documentations.

    You may speed-up the time to build notebook examples:

    .. code-block:: bash

        cd docs
        make clean
        NOTEBOOK_FAST_RUN=1 make html

    You may also skip the notebook examples altogether:

    .. code-block:: bash

        cd docs
        make clean
        NO_NOTEBOOKS=1 make html

.. tip::

    `VS Code <https://code.visualstudio.com/>`_ users may consider the
    `Live Preview <https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server>`_
    extension to auto-refresh the preview of the generated HTML pages after every ``make html``.


Wheel File
----------

Generate the ``.whl`` file as follow:

.. code-block:: bash

    python -m build --wheel --no-isolation

To clean-up the build artifacts:

.. code-block:: bash

    VIRTUAL_ENV='' python setup.py clean --all

To clean-up the build artifacts **and** your currently active virtual environment:

.. code-block:: bash

    python setup.py clean --all
