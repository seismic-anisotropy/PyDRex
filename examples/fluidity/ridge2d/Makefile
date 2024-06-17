# Make options to keep what's left of our sanity.
.SHELLFLAGS += -u
.ONESHELL:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
# Path options, SIM_NAME is the name of the .msh and .py files.
SIM_NAME := ridge2d
SRC_DIR := ridge2d
OUT_DIR := _out
# Number of processors to use for parallelisation.
NPROCS := 4
# Initial geometry options (in units of metres), passed to `pydrex-mesh`.
# Negative depth value makes z-values negative.
WIDTH := 1e6
DEPTH := -4e5
HALF_WIDTH := $(shell python3 -c 'print($(WIDTH)/2)')
HALF_DEPTH := $(shell python3 -c 'print($(DEPTH)/2)')
RESOLUTION_HI := 5e2
RESOLUTION_LO := 1e4
RESOLUTION_MED := 1e3
# Initial options for temperature field.
SURFACE_TEMP := 273
DIFF_TEMP := 1350
THERM_DIFFUSIVITY := 2.23e-6
THERM_FIT := "Standard"
MIN_AGE := 1
# Initial conditions, used to parametrise velocity fields and spawn particles.
# INIT_HORIZ can be a space-delimited array of values (for multiple particles).
# NOTE: PLATE_SPEED is in cm/yr here.
PLATE_SPEED := 2.0
# INIT_HORIZ := 3.13e4 9.74e4 2.02e5 3.97e5
# INIT_DEPTH := 2e5
# # Recrystallisation parameters.
# STRESS_EXPONENT := 1.5
# DEFORMATION_EXPONENT := 3.5
# GBM_MOBILITY := 10
# GBS_THRESHOLD := 0.3
# NUCLEATION_EFFICIENCY := 5

all: $(OUT_DIR)/$(SIM_NAME)_P$(NPROCS).flml
	@echo "********** Running fluidity in parallel with verbose logging enabled..."
	cd $(OUT_DIR) && mpiexec -np $(NPROCS) fluidity -v2 -l $<

$(OUT_DIR)/$(SIM_NAME)_P$(NPROCS).flml: $(OUT_DIR)/$(SIM_NAME).flml \
	$(OUT_DIR)/$(SIM_NAME).msh $(OUT_DIR)/$(SIM_NAME).py $(OUT_DIR)/$(SIM_NAME).ini
	@echo "********** Creating parallel flml file using flredecomp..."
	./envcheck.sh -f
	cd $(OUT_DIR) && mpiexec -np $(NPROCS) flredecomp -i 1 -o $(NPROCS) $(SIM_NAME) $(SIM_NAME)_P$(NPROCS)

$(OUT_DIR)/$(SIM_NAME).ini: $(OUT_DIR)/$(SIM_NAME).py
	@echo "********** Setting up initial conditions and recryst. parameters..."
	echo "[initial conditions]" > $@
	echo "WIDTH = $(WIDTH)" >> $@
	echo "DEPTH = $(DEPTH)" >> $@
	echo "HALF_WIDTH = $(HALF_WIDTH)" >> $@
	echo "HALF_DEPTH = $(HALF_DEPTH)" >> $@
	echo "RESOLUTION_HI = $(RESOLUTION_HI)" >> $@
	echo "RESOLUTION_LO = $(RESOLUTION_LO)" >> $@
	echo "RESOLUTION_MED = $(RESOLUTION_MED)" >> $@
	echo "SURFACE_TEMP = $(SURFACE_TEMP)" >> $@
	echo "DIFF_TEMP = $(DIFF_TEMP)" >> $@
	echo "THERM_DIFFUSIVITY = $(THERM_DIFFUSIVITY)" >> $@
	echo "THERM_FIT = $(THERM_FIT)" >> $@
	echo "MIN_AGE = $(MIN_AGE)" >> $@
	echo "PLATE_SPEED = $(PLATE_SPEED)" >> $@

$(OUT_DIR)/$(SIM_NAME).msh: $(OUT_DIR)/$(SIM_NAME).geo
	@echo "********** Building the mesh file..."
	./envcheck.sh -m
	gmsh -2 -o $@ $<

$(OUT_DIR)/$(SIM_NAME).py: $(SRC_DIR)/$(SIM_NAME).py
	@echo "********** Copying python field setters..."
	mkdir -p $(@D)
	cp -f $< $@

$(OUT_DIR)/$(SIM_NAME).flml: $(SRC_DIR)/$(SIM_NAME).flml
	@echo "********** Copying serial flml file..."
	mkdir -p $(@D)
	cp -f $< $@

$(OUT_DIR)/$(SIM_NAME).geo: $(SRC_DIR)/$(SIM_NAME).geo
	@echo "********* Copying the .geo file..."
	mkdir -p $(@D)
	sed -i "s/width = .*;/width = $(WIDTH);/" $<
	sed -i "s/depth = .*;/depth = $(DEPTH);/" $<
	sed -i "s/res_hi = .*;/res_hi = $(RESOLUTION_HI);/" $<
	sed -i "s/res_lo = .*;/res_lo = $(RESOLUTION_LO);/" $<
	sed -i "s/res_med = .*;/res_med = $(RESOLUTION_MED);/" $<
	cp -f $< $@

.PHONY: all clean
clean:
	rm -rf $(OUT_DIR)
