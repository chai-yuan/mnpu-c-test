NAME = tinyml-esc-cnn
TOP_DIR = $(realpath $(dir ../../))

CFLAGS += -I$(TOP_DIR)

ACCE ?= soft
MAIN ?= main.c

SRCS += $(MAIN)
SRCS += $(TOP_DIR)/tinyml/tinyml.c $(TOP_DIR)/tinyml/arch/$(ACCE).c
SRCS += $(shell find $(TOP_DIR)/arena/ -name "*.c")

include ../../../runtime/Makefile
