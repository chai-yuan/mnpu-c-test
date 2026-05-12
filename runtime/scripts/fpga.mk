CC       = riscv64-unknown-elf-gcc
OBJDUMP  = riscv64-unknown-elf-objdump
OBJCOPY  = riscv64-unknown-elf-objcopy

CFLAGS  += -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 -Wall -Wextra -MMD
CFLAGS  += -I$(RUNTIME_DIR)/tinylibc/include

FPGA_DIR = $(RUNTIME_DIR)/src/fpga

SRCS    += $(FPGA_DIR)/start.S
SRCS    += $(FPGA_DIR)/init.c
SRCS    += $(wildcard $(RUNTIME_DIR)/tinylibc/src/*.c)

LDFLAGS += -T$(FPGA_DIR)/fpga.ld -nostdlib

gen: $(IMAGE).bin
	@echo "  FPGA $(IMAGE).coe generated"
	@python $(FPGA_DIR)/gen-coe.py $(IMAGE).bin --output $(IMAGE).coe

.PHONY: gen
