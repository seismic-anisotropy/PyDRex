# PyDRex tests

Running the tests requires [pytest](https://docs.pytest.org).
From the root of the source tree, run `pytest`.
To print more verbose information (including INFO level logging),
use the flag `pytest -v`.
The custom optional flag `--outdir="OUT"` can be used
to produce output figures, data dumps and logs and save them in the directory `"OUT"`.
The value `"."` can be used to save these in the current directory.
Long tests/examples are disabled by default to prevent
clogging up the automatic testing suite, they can be enabled with `--runslow`.
To mark a test as slow,
add the `@pytest.mark.slow` decorator above its method definition.

Tests should not produce persistent output by default.
If a test method can produce such output for debugging or visualisation,
it should accept the `outdir` positional argument,
and check if its value is not `None`.
If `outdir is None` then no persistent output should be produced.
If `outdir` is a directory path (string):
- logs can be saved by using the `pydrex.logger.logfile_enable` context manager,
  which accepts a path name and an optional logging level as per Python's `logging` module
  (the default is `logging.DEBUG` which implies the most verbose output),
- figures can be saved by (implementing and) calling a helper from `pydrex.visualisation`, and
- data dumps can be saved to `outdir`, e.g. in `.npz` format (see the `pydrex.minerals.Mineral.save` method)
In all cases, saving to `outdir` should handle creation of parent directories.
To handle this as well as relative paths, we provide `pydrex.io.resolve_path`,
which is a thin wrapper around some `pathlib` methods.
