# 2D corner flow fluidity example

Most parameters can be set in the top of the makefile, except for the number of
grains, which is defined by `N - 21` where `N` is the dimension of the `CPO_`
array in the fluidity `.flml` file.

To run the example, ensure that the `envcheck.sh` script is available, and
simply run `make`. Alternatively, the makefile can be launched from a parent
directory using `make -f some_directory/Makefile`.
