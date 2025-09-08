SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXDOCS    ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
PACKAGE       = gradgraph
PACKAGEDIR    = ../

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help rst Makefile

rst:
	@$(SPHINXDOCS) -o "$(SOURCEDIR)" "$(PACKAGEDIR)" -e -M

%: Makefile rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm $(SOURCEDIR)/$(PACKAGE)*.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
