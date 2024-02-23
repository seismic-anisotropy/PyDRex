# Path options, SIM_NAME is the name of the .msh and .py files.
SIM_NAME := corner_flow_2d
SRC_DIR := cornerflow2d
OUT_DIR := _out
# Initial geometry options (in units of metres), passed to `meshgen.py`.
WIDTH := 1e6
DEPTH := 2e5
RES_HI := 1e2
RES_LO := 1e4
# Initial conditions, used to parametrise velocity fields and spawn particles.
# With PLATE_SPEED := 2 (cm/yr), INIT_HORIZ := 1e5 m, INIT_DEPTH := 1e5 m
# it takes about 4.9e14 seconds to traverse the domain.
# INIT_HORIZ can be a space-delimited array of values (for multiple particles).
PLATE_SPEED := 2.0
INIT_HORIZ := 3.13e4 9.74e4 2.02e5 3.97e5
INIT_DEPTH := 2e5
# Recrystallisation parameters.
STRESS_EXPONENT := 3.5
DEFORMATION_EXPONENT := 1.5
GBM_MOBILITY := 10
GBS_THRESHOLD := 0
NUCLEATION_EFFICIENCY := 5


all: $(OUT_DIR)/$(SIM_NAME).flml $(OUT_DIR)/$(SIM_NAME).msh $(OUT_DIR)/$(SIM_NAME).py $(OUT_DIR)/$(SIM_NAME).ini
	@echo ********** Running fluidity with verbose logging enabled...
	./envcheck.sh -f
	cd $(OUT_DIR) && fluidity -v2 -l $(SIM_NAME).flml

$(OUT_DIR)/$(SIM_NAME).ini: $(OUT_DIR)/$(SIM_NAME).py
	@echo ********** Setting up initial conditions and recryst. parameters...
	echo "[initial conditions]" > $@
	echo "WIDTH = $(WIDTH)" >> $@
	echo "DEPTH = $(DEPTH)" >> $@
	echo "RES_HI = $(RES_HI)" >> $@
	echo "RES_LO = $(RES_LO)" >> $@
	echo "PLATE_SPEED = $(PLATE_SPEED)" >> $@
	echo "INIT_HORIZ = $(INIT_HORIZ)" >> $@
	echo "INIT_DEPTH = $(INIT_DEPTH)" >> $@
	echo "STRESS_EXPONENT = $(STRESS_EXPONENT)" >> $@
	echo "DEFORMATION_EXPONENT = $(DEFORMATION_EXPONENT)" >> $@
	echo "GBM_MOBILITY = $(GBM_MOBILITY)" >> $@
	echo "GBS_THRESHOLD = $(GBS_THRESHOLD)" >> $@
	echo "NUCLEATION_EFFICIENCY = $(NUCLEATION_EFFICIENCY)" >> $@

$(OUT_DIR)/$(SIM_NAME).py: $(SRC_DIR)/$(SIM_NAME).py
	@echo ********** Copying python velocity callables...
	mkdir -p $(@D)
	cp -f $< $@

$(OUT_DIR)/$(SIM_NAME).flml: $(SRC_DIR)/$(SIM_NAME).flml
	@echo ********** Copying serial flml file...
	mkdir -p $(@D)
	cp -f $< $@

$(OUT_DIR)/$(SIM_NAME).msh: $(SRC_DIR)/meshgen.py
	@echo ********** Building the mesh file...
	./envcheck.sh -m
	python3 $< -o $@ -w $(WIDTH) -d $(DEPTH) -u $(RES_HI) -l $(RES_LO)
