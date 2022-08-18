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

# Minimal makefile for Sphinx documentation
#
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS      ?= -v
SPHINXBUILD     ?= sphinx-build
SPHINXAPIDOC    ?= sphinx-apidoc
SOURCEDIR       = .
BUILDDIR        = _build
APIDIR          = api
PKGDIR          = ../src/a2rl
SPHINXAUTOBUILD = sphinx-autobuild

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "  livehtml    to run live-preview (require Python package sphinx-autobuild)"

.PHONY: help Makefile prep

clean: Makefile
	@rm -fr "$(APIDIR)"/ "references/api/" "references/_api/" ../notebooks/model_*/ || true
	@rm $(BUILDDIR)/refdata auto-notebooks changelog.md || true
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) || true
	@rmdir "$(BUILDDIR)" || true

livehtml:
	@$(SPHINXAUTOBUILD) --re-ignore 'docs/README.md' "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

prep:
	@mkdir -p _build
	@ln -s ../CHANGELOG.md changelog.md
	@rm $(BUILDDIR)/refdata auto-notebooks || true
ifdef NO_NOTEBOOKS
	@echo Skipping notebooks...
else
	@ln -s ../notebooks auto-notebooks || true
endif

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile prep
	echo NOTEBOOK_FAST_RUN=$(NOTEBOOK_FAST_RUN)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
