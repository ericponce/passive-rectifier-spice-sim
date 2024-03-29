##############################################################################
#                                System Config                               #
##############################################################################

LTSPICE = /mnt/c/Program\ Files/LTC/LTspiceXVII/XVIIx64.exe

CIR_EXT = asc
RAW_EXT = raw
LOG_EXT = log
NET_EXT = net
OP_RAW_EXT = op.raw

##############################################################################
#                                 Sim Config                                 #
##############################################################################

SIM_NAME = base

OUT_DIR = build

##############################################################################
#                                    Flags                                   #
##############################################################################

LTSPICE_ARGS = -Run -b

##############################################################################
#                                 Dependencies                               #
##############################################################################

VPATH = ss_models env_models


##############################################################################
#                                  Formatting                                #
##############################################################################

OUT_DIR_F = $(strip $(OUT_DIR))/

SPICE_RAW_FILE = $(SIM_NAME).$(RAW_EXT)
SPICE_LOG_FILE = $(SIM_NAME).$(LOG_EXT)
SPICE_NET_FILE = $(SIM_NAME).$(NET_EXT)
SPICE_OP_RAW_FILE = $(SIM_NAME).$(OP_RAW_EXT)

SPICE_RAW_FILE_B = $(OUT_DIR_F)$(SPICE_RAW_FILE)

SPICE_OUTPUT_FILES = $(SPICE_RAW_FILE) $(SPICE_LOG_FILE) $(SPICE_NET_FILE) $(SPICE_OP_RAW_FILE)
SPICE_OUTPUT_FILES_B = $(addprefix $(OUT_DIR_F), $(notdir $(SPICE_OUTPUT_FILES)))

##############################################################################
#                                   Recipes                                  #
##############################################################################

all : make_output_dir $(SPICE_RAW_FILE_B)

$(OUT_DIR_F)%.raw : %.asc %.gen make_output_dir
	@$(LTSPICE) $(LTSPICE_ARGS) $<
	@mv $(dir $<)$(notdir $@) $@
	@mv $(dir $<)$(notdir $(basename $@)).$(LOG_EXT) $(basename $@).$(LOG_EXT)
	@mv $(dir $<)$(notdir $(basename $@)).$(NET_EXT) $(basename $@).$(NET_EXT)
	@mv $(dir $<)$(notdir $(basename $@)).$(OP_RAW_EXT) $(basename $@).$(OP_RAW_EXT)

$(SPICE_RAW_FILE_B) : Makefile


##############################################################################
#                               Utility Recipes                              #
##############################################################################

run :
	python sim.py

make_output_dir :
	$(shell mkdir $(OUT_DIR) 2>/dev/null)
	$(shell mkdir $(OUT_DIR)/ss_models 2>/dev/null)
	$(shell mkdir $(OUT_DIR)/env_models 2>/dev/null)


clean :
	rm *.gen *.$(RAW_EXT) *.$(LOG_EXT) *.$(NET_EXT) *.$(OP_RAW_EXT)
	rm $(OUT_DIR_F)*.$(RAW_EXT)
	rm $(OUT_DIR_F)*.$(LOG_EXT)
	rm $(OUT_DIR_F)*.$(NET_EXT)
	rm $(OUT_DIR_F)*.$(OP_RAW_EXT)

.PHONY : all run clean