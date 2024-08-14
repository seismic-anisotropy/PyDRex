.SHELLFLAGS += -u
.ONESHELL:
MAKEFLAGS += --no-builtin-rules
VERSION = $(shell python -m setuptools_scm -f plain)

# NOTE: Keep this at the top! Bare `make` should just build the sdist + bdist.
dist:
	python -m build

release: dist
	python -m twine upload dist/*

test:
	mkdir out
	pytest -v --outdir=out

# WARNING: --math fetches .js code from a CDN, be careful where it comes from:
# https://github.com/mitmproxy/pdoc/security/advisories/GHSA-5vgj-ggm4-fg62
html:
	2>pdoc.log pdoc -t docs/template -o html pydrex !pydrex.mesh !pydrex.distributed tests \
		--favicon "https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/favicon32.png" \
		--footer-text "PyDRex $(VERSION)" \
		--math

# WARNING: --math fetches .js code from a CDN, be careful where it comes from:
# https://github.com/mitmproxy/pdoc/security/advisories/GHSA-5vgj-ggm4-fg62
live_docs:
	pdoc -t docs/template pydrex !pydrex.mesh !pydrex.distributed tests \
		--favicon "https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/favicon32.png" \
		--footer-text "PyDRex $(VERSION)" \
		--math

clean:
	rm -rf dist
	rm -rf out
	rm -rf html
	rm -rf pdoc.log

.PHONY: release test live_docs clean
