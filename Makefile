.SHELLFLAGS += -u
.ONESHELL:
MAKEFLAGS += --no-builtin-rules

dist:
	python -m build

release: dist
	python -m twine upload dist/*

test:
	mkdir out
	pytest -v --outdir=out

clean:
	rm -r dist
	rm -r out

.PHONY: release test clean
