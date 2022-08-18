# Build documentations <!-- omit in toc -->

**IMPORTANT**:
*Unless stated otherwise, all the instructions on this page assumes `GITROOT/` as your current
directory.*

## 1. Pre-Requisites

The Python packages required to build the documentations can be installed as follows:

```bash
pip install -r <(cat requirements.txt requirements-docs.txt)
```

In addition, you need [pandoc](https://www.pandoc.org) to build some of the documentations written
as Jupyter notebooks (i.e., `docs/**/*.ipynb` files). To install on your computer, please refer to
[Pandoc's installation guide](https://pandoc.org/installing.html).

You also need [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) to build
[Mermaid](https://mermaid-js.github.io/) diagrams. To install on your computer, please refer to
its installation instructions. **NOTE**: this is a workaround until this
[issue](https://github.com/mgaitan/sphinxcontrib-mermaid/issues/74) is fixed.

## 2. Generate HTML documentations

Run these commands:

```bash
cd docs
make clean
make html
```

then point your web browser to `docs/_build/html/index.html`.

When writing documentations, you may speed-up the build time with fewer epochs in the notebook
examples:

```bash
cd docs
make clean
NOTEBOOK_FAST_RUN=1 make html
```

You may even skip the notebook examples to further speed-up the build time, especially working with
documentations outside the notebook examples:

```bash
cd docs
make clean
NO_NOTEBOOKS=1 make html
```

## 3. Development

[VS Code](https://code.visualstudio.com/) users may consider the
[Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) extension
to auto-refresh the preview of the generated HTML pages after every ``make html``.

## 4. FAQ

### 4.1. Linebreaks in Mermaid Sequence Diagram

**TL;DR:** Please use `<div></div>` instead of `<br>`.

**Longer explanation:** [nbsphinx](https://github.com/spatialaudio/nbsphinx) has a conflict
with [sphinxcontrib-mermaid](https://github.com/mgaitan/sphinxcontrib-mermaid) (see this
[issue](https://github.com/mgaitan/sphinxcontrib-mermaid/issues/74)). This requires Mermaid diagrams
to be [converted](https://github.com/mgaitan/sphinxcontrib-mermaid/issues/74#issue-887462744) to SVG
using [mermaid-cli](https://github.com/mermaid-js/mermaid-cli). Unfortunately, `mermaid-cli` has a
known [issue](https://github.com/mermaid-js/mermaid/issues/384) for choking on `<br>`, `<br/>`, or
`<br />`. The [workaround](https://github.com/mermaid-js/mermaid/issues/384#issuecomment-366119634)
is to use `<div></div>` instead.

### 4.2. Auto-build during Development

You may use [sphinx-autobuild](https://github.com/executablebooks/sphinx-autobuild) to automatically
build and live-preview the documentations.

> **BEWARE**: as of this writing,
> [sphinx-autobuild](https://github.com/executablebooks/sphinx-autobuild) runs a ``make`` command
> for every file modified (i.e., saved). This may be too excessive when when multiple files are
> saved in a short period of time, e.g., by your favorite IDE, editor, pre-commit hooks, etc.

```bash
# Assume requirements-docs.txt has been installed
cd GITROOT/docs
pip install sphinx-autobuild
make livehtml
```

and you should see this kind of messages in your terminal:

```text
...

The HTML pages are in _build.
[I 220405 18:25:21 server:335] Serving on http://127.0.0.1:8000
[I 220405 18:25:21 handlers:62] Start watching changes
[I 220405 18:25:21 handlers:64] Start detecting changes
```

The line `Serving on http://127.0.0.1:8000` indicates that the live-preview server is running at
this address. You can then point your web-browser to <http://127.0.0.1:8000>, and it will
auto-refresh whenever you make changes to the `.rst` files.

To stop the live-preview server, press `Ctrl-C` on the same terminal where you run `make livehtml`.
