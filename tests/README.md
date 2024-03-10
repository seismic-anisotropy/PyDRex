# PyDRex tests

Running the tests requires [pytest](https://docs.pytest.org).
From the root of the source tree, run `pytest`.
To print more verbose information (including INFO level logging),
such as detailed test progress, use the flag `pytest -v`.
The custom optional flag `--outdir="OUT"` is recommended
to produce output figures, data dumps and logs and save them in the directory `"OUT"`.
The value `"."` can be used to save these in the current directory.

Running individual tests or test subsets is possible using the pytest
`-k="<pattern>"` command line flag, which accepts a string pattern that is
matched against the names of test classes or methods.
To see a full list of available tests, use the command `pytest --co`. This
produces a rather long list and it is recommended to view the output with a
pager like `less` on Linux.

In total, the following custom pytest command line flags are defined by PyDRex:
- `--outdir` (described above)
- `--runbig` (enable tests which require a large amount of RAM)
- `--runslow` (enable slow tests which require HPC resources, implies `--runbig`)
- `--ncpus` (number of CPU cores to use for shared memory multiprocessing, set to one less than the available maximum by default)
- `--fontsize` (Matplotlib `rcParams["font.size"]`)
- `--markersize` (Matplotlib `rcParams["lines.markersize"]`)
- `--linewidth` (Matplotlib `rcParams["lines.linewidth"]`)

Tests which require a “significant” amount of memory (> ~16GB RAM) are disabled by default.
To fully check the functionality of the code, it is recommended to run these locally
by using the `--runbig` flag before moving to larger simulations.

Long tests/examples are also disabled by default and can be enabled with `--runslow`.
It is recommended to run these on a HPC cluster infrastructure (>100GB RAM, >32 cores).
The number of cores to use for shared memory multiprocessing can be specified with `--ncpus`.

## Writing tests

- To mark a test as “big” (i.e. requiring more than ~16GB RAM), apply the
  `@pytest.mark.big` decorator to the corresponding method definition.

- To mark a test as “slow” (i.e. requiring more than ~32 cores), apply the
  `@pytest.mark.slow` decorator to the corresponding method definition.

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
