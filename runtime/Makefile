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

RUNTIME_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(RUNTIME_DIR)scripts/$(ARCH).mk

SRCS 	+= $(wildcard $(RUNTIME_DIR)/lib/*.c)
CFLAGS  += -I$(PROJ_DIR) -I$(RUNTIME_DIR)
IMAGE   := $(DST_DIR)/$(NAME)

OBJS := $(addprefix $(DST_DIR)/, $(addsuffix .o, $(basename $(SRCS))))

$(DST_DIR)/%.o: %.c
	@mkdir -p $(dir $@) && echo "  CC    $<"
	@$(CC) -std=c11 $(CFLAGS) -c -o $@ $(realpath $<)

$(IMAGE): $(OBJS)
	@echo "  LD    $(notdir $(IMAGE))"
	@$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

all: $(IMAGE)
.PHONY: all

clean:
	rm -rf $(PROJ_DIR)/build/
.PHONY: clean

-include $(addprefix $(DST_DIR)/, $(addsuffix .d, $(basename $(SRCS))))
