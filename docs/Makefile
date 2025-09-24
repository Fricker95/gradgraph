SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXDOCS    ?= sphinx-apidoc
TEMPLATEDIR   ?= _templates
SOURCEDIR     = source
BUILDDIR      = build
PACKAGE       = gradgraph
PACKAGEDIR    = ../

.PHONY: help clean rst Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm $(SOURCEDIR)/$(PACKAGE)*.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

rst:
	@$(SPHINXDOCS) -o "$(SOURCEDIR)" "$(PACKAGEDIR)" -e -M --no-toc

# -f -t $(SOURCEDIR)/$(TEMPLATEDIR)

%: Makefile rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
