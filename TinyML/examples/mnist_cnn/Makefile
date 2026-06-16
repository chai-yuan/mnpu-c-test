NAME = tinyml-mnist-cnn
TOP_DIR = $(realpath $(dir ../../))

CFLAGS += -I$(TOP_DIR)

SRCS += main.c
SRCS += $(shell find $(TOP_DIR)/tinyml/ -name "*.c")
SRCS += $(shell find $(TOP_DIR)/arena/ -name "*.c")

include ../../../runtime/Makefile
