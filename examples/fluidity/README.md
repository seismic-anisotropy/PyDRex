# Fluidity examples

Most parameters can be set in the relevant makefiles, except for the number of
grains, which is defined by `(N - 22) / 10` where `N` is the dimension of the
`CPO_` array in the Fluidity `.flml` file. Timestepping options are also set
in the `.flml` file.

To run the example contained in `example_directory`,
ensure that the `envcheck.sh` script is available, and simply run
`make -f example_directory/Makefile`.
To clean previous simulation output use the
`make -f example_directory/Makefile clean` target.

This table lists the corresponding PyDRex tests:

| Fluidity example | PyDRex test |
| --- | --- |
| `advection2d` | `test_simple_shear_2d.TestOlivineA.test_dudz_pathline` |
| `corner2d` | `test_corner_flow_2d.TestOlivineA.test_steady4` |
