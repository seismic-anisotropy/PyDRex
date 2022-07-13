# PyDRex tests

Running the tests requires [pytest](https://docs.pytest.org).
From the root of the source tree, run `pytest`.
The custom optional flag `--outdir="OUT"` can be used
to produce output figures and save them in the directory `"OUT"`.
The value `"."` can be used to save figures in the current directory.

Tests should not produce figures by default.
If a test method can produce figures, it should accept the `outdir`
positional argument, and check if its value is not `None`.
If `outdir is None` then no figures should be produced.
This also applies to any other large outputs like verbose logs or data dumps.
