ifeq ($(NAME),)
  $(error NAME is not set)
endif
ifeq ($(SRCS),)
  $(error SRCS is not set)
endif

ARCH ?= native
$(info # Building $(NAME) [$(ARCH)])

PROJ_DIR := $(shell pwd)
DST_DIR  := $(PROJ_DIR)/build/$(ARCH)
$(shell mkdir -p $(DST_DIR))

RUNTIME_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

SRCS 	+= $(wildcard $(RUNTIME_DIR)/lib/*.c)
CFLAGS  += -I$(PROJ_DIR) -I$(RUNTIME_DIR)
LDFLAGS +=

IMAGE   := $(DST_DIR)/$(NAME)
DUMP    := $(IMAGE).dump
BIN     := $(IMAGE).bin

CC 	    ?= gcc
OBJDUMP ?= objdump
OBJCOPY ?= objcopy

all: $(IMAGE) $(DUMP) $(BIN)
.PHONY: all

include $(RUNTIME_DIR)/scripts/$(ARCH).mk

OBJS := $(addprefix $(DST_DIR)/, $(addsuffix .o, $(basename $(SRCS))))

$(DST_DIR)/%.o: %.c
	@mkdir -p $(dir $@) && echo "  CC    $<"
	@$(CC) -std=c11 $(CFLAGS) -c -o $@ $(realpath $<)

$(DST_DIR)/%.o: %.S
	@mkdir -p $(dir $@) && echo "  AS    $<"
	@$(CC) $(CFLAGS) -c -o $@ $(realpath $<)

$(IMAGE): $(OBJS)
	@echo "  LD    $(notdir $(IMAGE))"
	@$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(DUMP): $(IMAGE)
	@echo "  DUMP  $(notdir $@)"
	@$(OBJDUMP) -d $< > $@

$(BIN): $(IMAGE)
	@echo "  BIN   $(notdir $@)"
	@$(OBJCOPY) -O binary $< $@

clean:
	rm -rf $(PROJ_DIR)/build/
.PHONY: clean

-include $(addprefix $(DST_DIR)/, $(addsuffix .d, $(basename $(SRCS))))
