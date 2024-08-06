.SHELLFLAGS += -u
.ONESHELL:
MAKEFLAGS += --no-builtin-rules
VERSION = $(shell git describe)

# NOTE: Keep this at the top! Bare `make` should just build the sdist + bdist.
dist:
	python -m build

release: dist
	python -m twine upload dist/*

test:
	mkdir out
	pytest -v --outdir=out

html:
	2>pdoc.log pdoc -t docs/template -o html pydrex !pydrex.mesh !pydrex.distributed tests \
		--favicon "https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/favicon32.png" \
		--footer-text "PyDRex $(VERSION)"

clean:
	rm -rf dist
	rm -rf out
	rm -rf html
	rm -rf pdoc.log

.PHONY: release test clean
