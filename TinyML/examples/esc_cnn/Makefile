NAME = tinyml-esc-cnn
TOP_DIR = $(realpath $(dir ../../))

CFLAGS += -I$(TOP_DIR)

ACCE ?= soft

SRCS += main.c
SRCS += $(TOP_DIR)/tinyml/tinyml.c $(TOP_DIR)/tinyml/arch/$(ACCE).c
SRCS += $(shell find $(TOP_DIR)/arena/ -name "*.c")

include ../../../runtime/Makefile
