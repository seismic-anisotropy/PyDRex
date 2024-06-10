# Make options to keep what's left of our sanity.
.SHELLFLAGS += -u
.ONESHELL:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
# Path options, SIM_NAME is the name of the .msh and .py files.
SIM_NAME := advection2d
SRC_DIR := advection2d
OUT_DIR := _out
# Initial geometry options (in units of metres), passed to `pydrex-mesh`.
WIDTH := 4e5
DEPTH := 4e5
RESOLUTION := 1e3
# Initial conditions, used to parametrise velocity fields and spawn particles.
# INIT_HORIZ can be a space-delimited array of values (for multiple particles).
STRAIN_RATE := 1e-15
INIT_HORIZ := -1e5
INIT_VERT := 1e5
# Recrystallisation parameters.
STRESS_EXPONENT := 1.5
DEFORMATION_EXPONENT := 3.5
GBM_MOBILITY := 10
GBS_THRESHOLD := 0.3
NUCLEATION_EFFICIENCY := 5

all: $(OUT_DIR)/$(SIM_NAME).flml \
	$(OUT_DIR)/$(SIM_NAME).msh $(OUT_DIR)/$(SIM_NAME).py $(OUT_DIR)/$(SIM_NAME).ini
	@echo "********** Running fluidity with verbose logging enabled..."
	./envcheck.sh -f
	cd $(OUT_DIR) && fluidity -v2 -l $(SIM_NAME).flml

$(OUT_DIR)/$(SIM_NAME).ini: $(OUT_DIR)/$(SIM_NAME).py
	@echo "********** Setting up initial conditions and recryst. parameters..."
	echo "[initial conditions]" > $@
	echo "STRAIN_RATE = $(STRAIN_RATE)" >> $@
	echo "INIT_HORIZ = $(INIT_HORIZ)" >> $@
	echo "INIT_VERT = $(INIT_VERT)" >> $@
	echo "STRESS_EXPONENT = $(STRESS_EXPONENT)" >> $@
	echo "DEFORMATION_EXPONENT = $(DEFORMATION_EXPONENT)" >> $@
	echo "GBM_MOBILITY = $(GBM_MOBILITY)" >> $@
	echo "GBS_THRESHOLD = $(GBS_THRESHOLD)" >> $@
	echo "NUCLEATION_EFFICIENCY = $(NUCLEATION_EFFICIENCY)" >> $@

$(OUT_DIR)/$(SIM_NAME).py: $(SRC_DIR)/$(SIM_NAME).py
	@echo "********** Copying python velocity callables..."
	mkdir -p $(@D)
	cp -f $< $@

$(OUT_DIR)/$(SIM_NAME).flml: $(SRC_DIR)/$(SIM_NAME).flml
	@echo "********** Copying serial flml file..."
	mkdir -p $(@D)
	cp -f $< $@

$(OUT_DIR)/$(SIM_NAME).msh:
	@echo "********** Building the mesh file..."
	./envcheck.sh -m
	pydrex-mesh -k="rectangle" -a xy $(WIDTH),$(DEPTH) -r G:$(RESOLUTION) $@

.PHONY: all clean
clean:
	rm -rf $(OUT_DIR)
