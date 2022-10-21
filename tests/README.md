# PyDRex tests

Running the tests requires [pytest](https://docs.pytest.org).
From the root of the source tree, run `pytest`.
Console logging is enabled by default (at `INFO` level),
however the pytest live logging level can be reduced
with the normal `pytest --log-cli-level` flag, e.g. `pytest --log-cli-level=WARNING`
The custom optional flag `--outdir="OUT"` can be used
to produce output figures, data dumps and logs and save them in the directory `"OUT"`.
The value `"."` can be used to save these in the current directory.

Tests should not produce persistent output by default.
If a test method can produce such output for debugging, it should accept the `outdir`
positional argument, and check if its value is not `None`.
If `outdir is None` then no persistent output should be produced.
If `outdir` is a directory path (string):
- logs can be saved by calling `pydrex.logger.logfile_enable`,
  which accepts a path name and an optional logging level as per Python's `logging` module
  (the default is `logging.DEBUG` which implies the most verbose output),
- figures can be saved by (implementing and) calling a helper from `pydrex.visualisation`, and
- data dumps can be saved to `outdir`, e.g. in `.npz` format (see the `Mineral.save` method)
In all cases, saving to `outdir` should handle creation of parent directories,
using the `pathlib` module:

```
pathlib.Path(f"{outdir}/some.file").parent.mkdir(parents=True, exist_ok=True)
```
