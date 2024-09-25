VERSION := `python -m setuptools_scm -f plain`
TAGS := `git tag --sort=-committerdate --format='%(refname:short)'|grep -v rc`
TAG_LATEST := `git tag --sort=-committerdate --format='%(refname:short)'|grep -v rc|tail -1`

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

all_docs:
    # WARNING: --math fetches .js code from a CDN, be careful where it comes from:
    # https://github.com/mitmproxy/pdoc/security/advisories/GHSA-5vgj-ggm4-fg62
    for tag in {{TAGS}}; do \
        git checkout "${tag}"; \
        rm requirements.txt; \
        ./tools/venv_install.sh -u; \
        echo "Building documentation for version ${tag}"; \
        pdoc -t docs/template -o "html/${tag}" pydrex !pydrex.mesh !pydrex.distributed tests \
            --favicon "https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/favicon32.png" \
            --footer-text "PyDRex $(python -m setuptools_scm -f plain)" \
            --math; \
    done
    ln -s html/{{TAG_LATEST}}/index.html html/index.html
    git checkout main
    rm requirements.txt
    ./tools/venv_install.sh -u

clean:
    rm -rf dist
    rm -rf out
    rm -rf html
    rm -rf pdoc.log
