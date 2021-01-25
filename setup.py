"""Setuptools shim for pre-PEP571 build methods."""
import setuptools  # type: ignore

# DO NOT EDIT, use setup.cfg instead, see:
# - https://setuptools.readthedocs.io/en/latest/userguide/declarative_config.html
# - https://docs.python.org/3/distutils/configfile.html
if __name__ == "__main__":
    setuptools.setup()
