r"""

#### Simulate crystallographic preferred orientation evolution in polycrystals

---

**This software is currently in early development.
Only modules that have tests are anywhere close to being usable.**

---

## About

The core routines are based on the original implementation by Édouard Kaminski,
which can be [downloaded from this link (~90KB)](http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz).
The reference paper is [Kaminski & Ribe 2001](https://doi.org/10.1016/s0012-821x(01)00356-9),
and an open-acess paper which discusses the model is [Fraters & Billen 2021](https://doi.org/10.1029/2021gc009846).

The package is currently not available on PyPi,
and must be installed by cloning the source code and using `pip install .`
(with the dot) in the top-level folder.
Multiprocessing is not yet available in the packaged version.
Running the tests requires [pytest](https://docs.pytest.org/en/stable/),
and the custom pytest flag `--outdir="OUT"` can be used to save output figures
to the folder called OUT (or the current folder, using `"."`).

## The D-Rex kinematic CPO model

The D-Rex model is used to compute crystallographic preferred orientation (CPO)
for polycrystals deforming by dislocation creep and dynamic recrystallization.
Polycrystals are discretized into "grains" which represent fractional volumes
of the total crystal that are characterised by a particular crystal lattice
orientation. For numerical efficiency, the number of grains in the model does
not change, and should only be interpreted as an approximation of the number
of physical grains. Dynamic recrystallization is modelled using statistical
expressions which approximate the interaction of each grain with an effective
medium based on the averaged dislocation energy of all other grains.
Note that the model is not suited to situations where static recrystallization
processes are significant.

The primary microphysical mechanism for plastic deformation of the polycrystal
is dislocation creep, which involves dislocation glide ("slip") along symmetry
planes of the mineral and dislocation climb, which allows for dislocations to
annihilate each other so that the number of dislocations reaches a steady-state.
The D-Rex model does not simulate dislocation climb, but implicitly assumes that
the dislocations are in steady-state so that the dislocation density of the
crystal can be described by

$$
ρ ∝ b^{-2} \left(\frac{σ}{μ}\right)^{p}
$$

where $b$ is the length of the Burgers' vector, $σ$ is the stress
and $μ$ is the shear modulus. The value of the exponent $p$ is given by the
'stress_exponent' input parameter. For an overview of available parameters,
see [the source code, for now...]

The effects of dynamic recrystallization are twofold. Grains with a higher than
average dislocation density may be affected by either grain nucleation, which is
the formation of initially small, strain-free sub-grains, or grain boundary
migratiton, by which process other grains of lower strain energy annex a portion
of its volume. Nucleation occurs mostly in grains oriented favourably for
dislocation glide, and the new grains also grow by grain boundary migration.
If nucleation is too inefficient, the dislocation density in deformation-aligned
grains will remain high and these grains will therefore shrink in volume. On the
other hand, if grain boundaries are too immobile, then nucleated grains will take
longer to grow, reducing the speed of CPO development and re-orientation.
Because nucleated grains are assumed to inherit the orientation of the parent,
they do not affect the model except by reducing the average dislocation density.
A grain boundary mobility parameter of $M^{∗} = 0$ will therefore disable any
recrystallization effects. Finally, the process of grain boundary sliding can
also be included, which simply disallows rotation of grains with very small volume.
This only affects CPO evolution by introducing a latency for the onset of grain
boundary migration in nucleated grains. It also manifests as an upper bound on
texture strength.

## Parameter reference

Model parameters should be provided in an `.ini` file.
The file must contain section headers enclosed by square braces
and key-value assignments, for example:

```
[Output]
olivine = volume_distribution, orientations

[D-Rex]
olivine_fraction = 1
stresss_exponent = 3.5
...
```

The following reference describes the available sections and their parameters.

### Geometry

This section allows for specifying the geometry of the model domain,
including any interpolation meshes.

| Parameter | Description |
| ---       | ---
| `meshfile`  | [not implemented]

### Output

Parameters in the output section control which variables are stored to files,
as well as any options for automatic postprocessing.

| Parameter | Description | Default
| ---       | ---         | ---
| `simulation_name` | a short name (without spaces) used for the output folder and metadata | `pydrex_example`
| `olivine` | the choice of olivine mineral outputs, from {`volume_distribution`, `orientations`} with multiple choices separated by a comma | `volume_distribution,orientations`
| `enstatite` | the choice of enstatite mineral outputs, from {`volume_distribution`, `orientations`} with multiple choices separated by a comma | `volume_distribution,orientations`

### D-Rex

Parameters in the D-Rex section specify the runtime configuration for the D-Rex model.
Read [the D-Rex introduction section](#the-d-rex-kinematic-cpo-model) for more details.

| Parameter | Description | Default
| ---       | ---         | ---
| `olivine_fraction` | the volume ratio for olivine compared to the total aggregate volume | `1` (no enstatite component)
| `stress_exponent` | the stress exponent $p$ that characterises the relationship between dislocation density and stress | `3.5`
| `gbm_mobility` | the dimensionless grain boundary mobility $M^{∗}$ which controls the chance for growth of grains with lower than average dislocation energy | `125`
| `gbs_threshold` | a threshold ratio of current to original volume below which small grains move by sliding rather than rotation | `0.3`
| `nucleation_efficiency` | the dimensionless nucleation efficiency which controls the chance for new, small, strain-free sub-grains to be created inside high dislocation energy grains | `5`
| `olivine_fabric` | [not implemented] | `A`
| `number_of_grains` | the number of initial grains per crystal| `1000`

"""
