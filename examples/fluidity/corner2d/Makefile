# Make options to keep what's left of our sanity.
.SHELLFLAGS += -u
.ONESHELL:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
# Path options, SIM_NAME is the name of the .msh and .py files.
SIM_NAME := corner2d
SRC_DIR := corner2d
OUT_DIR := _out
# Initial geometry options (in units of metres), passed to `pydrex-mesh`.
WIDTH := 1e6
DEPTH := 2e5
HALF_WIDTH := $(shell python3 -c 'print($(WIDTH)/2)')
HALF_DEPTH := $(shell python3 -c 'print($(DEPTH)/2)')
RESOLUTION_HI := 1e2
RESOLUTION_LO := 1e4
# Initial conditions, used to parametrise velocity fields and spawn particles.
# INIT_HORIZ can be a space-delimited array of values (for multiple particles).
# NOTE: PLATE_SPEED is in cm/yr here.
PLATE_SPEED := 2.0
INIT_HORIZ := 3.13e4
INIT_DEPTH := 2e5
# Recrystallisation parameters.
STRESS_EXPONENT := 1.5
DEFORMATION_EXPONENT := 3.5
GBM_MOBILITY := 10
GBS_THRESHOLD := 0.3
NUCLEATION_EFFICIENCY := 5

$(OUT_DIR)/fluidity.log-0: $(OUT_DIR)/$(SIM_NAME).flml \
	$(OUT_DIR)/$(SIM_NAME).msh $(OUT_DIR)/$(SIM_NAME).py $(OUT_DIR)/$(SIM_NAME).ini
	@echo "********** Running fluidity with verbose logging enabled..."
	./envcheck.sh -f
	cd $(OUT_DIR) && fluidity -v2 -l $(SIM_NAME).flml

$(OUT_DIR)/$(SIM_NAME).ini: $(OUT_DIR)/$(SIM_NAME).py
	@echo "********** Setting up initial conditions and recryst. parameters..."
	echo "[initial conditions]" > $@
	echo "PLATE_SPEED = $(PLATE_SPEED)" >> $@
	echo "INIT_HORIZ = $(INIT_HORIZ)" >> $@
	echo "INIT_DEPTH = $(INIT_DEPTH)" >> $@
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
	pydrex-mesh -k="rectangle" -a xy $(WIDTH),$(DEPTH) \
		-c $(HALF_WIDTH),-$(HALF_DEPTH) -r NW:$(RESOLUTION_HI),SE:$(RESOLUTION_LO) $@

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
