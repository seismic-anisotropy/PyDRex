VERSION := `python -m setuptools_scm -f plain`

build:
    python -m build

release: build
    python -m twine upload dist/*

test:
    mkdir out
    pytest -v --outdir=out

html:
    # WARNING: --math fetches .js code from a CDN, be careful where it comes from:
    # https://github.com/mitmproxy/pdoc/security/advisories/GHSA-5vgj-ggm4-fg62
    2>pdoc.log pdoc -t docs/template -o html pydrex !pydrex.mesh !pydrex.distributed tests \
        --favicon "https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/favicon32.png" \
        --footer-text "PyDRex {{VERSION}}" \
        --math

live_docs:
    # WARNING: --math fetches .js code from a CDN, be careful where it comes from:
    # https://github.com/mitmproxy/pdoc/security/advisories/GHSA-5vgj-ggm4-fg62
    pdoc -t docs/template pydrex !pydrex.mesh !pydrex.distributed tests \
        --favicon "https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/favicon32.png" \
        --footer-text "PyDRex {{VERSION}}" \
        --math

clean:
    rm -rf dist
    rm -rf out
    rm -rf html
    rm -rf pdoc.log
