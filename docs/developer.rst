Developer Guide
===============

We welcome and encourage all kind of help, such as bug reports, bug fixes, documentation
improvements, enhancements, ideas, marketing and communications, devops, project coordination, etc.

.. important::

    Unless stated otherwise, all the instructions on this page assumes ``GITROOT/`` as your current
    directory.

Pre-Requisites
--------------

The Python packages required for development is specified in ``requirements*.txt``. Please
install those dependencies as follows:

.. code-block::

    pip install -r <(cat requirements.txt requirements-*.txt)



Additional Dependencies
-----------------------

==========================  =============================================
``requirements-test.txt``   Required for unit tests
``requirements-docs.txt``   Required to generate the html documentations.
``requirements-build.txt``  Required to build the `wheel <https://packaging.python.org/en/latest/glossary/#term-wheel>`_.
``requirements-dev.txt``    Recommended packages during development.
==========================  =============================================


Advance Setup
-------------

Perform an editable install so that your changes to the code base will be reflected automatically.

.. code-block:: bash

    git clone git@github.com:awslabs/amazon-accessible-rl-sdk.git
    cd amazon-accessible-rl-sdk
    pip install -e .

Hygiene Practice
----------------

.. important:: **Insist on the highest standards**

    Leaders have relentlessly high standards --- many people may think these standards are
    unreasonably high. Leaders are continually raising the bar and drive their teams to deliver high
    quality products, services, and processes. Leaders ensure that defects do not get sent down the
    line and that problems are fixed so they stay fixed.

To ensure high-quality merge requests (MR) and shorten the code-review cycle, you're strongly
recommended to perform these tasks before creating an MR.

Although the project repository also runs the same checks as CI on your MR as a pre-cautionary
measure, you're still strongly recommended to run these tasks locally to quickly catch and fix
low-hanging-fruit violations.

Linters
~~~~~~~

The code-base comes with a set of git pre-commit hooks listed in the ``.pre-commit-config.yaml``
file.

Unit Tests
~~~~~~~~~~

Unit tests are done using the `pytest <https://docs.pytest.org/en/stable/>`_ framework.

We require that your contributions do not break existing tests. And if you found the existing tests
are buggy, please contribute your fixes.

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

    This project takes a brutally pragmatic stance on
    `type checks <https://docs.python.org/3/library/typing.html>`_. We acknowledge that type checks
    is still a relatively new concept (and coding habit), hence not all contributers agree to it or
    have it build into their muscle memory. More over, we may need to deal with some Python
    dependencies with varying degree of type checks.

    As such, the continuous integration (CI) still allows MR to fail type checks. Nevertheless,
    you're still highly recommended to do your best to correctly implement type checks in your
    code contributions.

    If you're new to type checks, we encourage you to learn more about its benefits and how-to on
    `mypy <https://mypy.readthedocs.io/en/stable/>`_ and
    `Python official documentation <https://peps.python.org/pep-0484/>`_.

.. warning::

    Although at present the CI process runs type checks only as an FYI basis, the long-term plan is
    to make it mandatory.

    As such, we highly recommend that you start to incorporate type checks from now, so you'll be
    quickly familiarized, and to shorten the code-review cycle by maintainers who, at times, might
    be a little bit pedantic |:slight_smile:|.

As a pre-requisite, you need to enable type hints on your pandas installations:

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
