CC       = riscv64-unknown-elf-gcc
OBJDUMP  = riscv64-unknown-elf-objdump
OBJCOPY  = riscv64-unknown-elf-objcopy

CFLAGS  += -march=rv32imf -mabi=ilp32 -nostdlib -ffreestanding -O2 -Wall -Wextra -MMD
CFLAGS  += -I$(RUNTIME_DIR)/tinylibc/include

SPIKE_DIR = $(RUNTIME_DIR)/src/spike

SRCS    += $(SPIKE_DIR)/start.S
SRCS    += $(SPIKE_DIR)/init.c
SRCS    += $(wildcard $(RUNTIME_DIR)/tinylibc/src/*.c)

LDFLAGS += -T$(SPIKE_DIR)/spike.ld -nostdlib

SPIKE = spike
SPIKE_FLAGS = --isa=rv32imf

run: $(IMAGE)
	@echo "  SPIKE $(notdir $(IMAGE))"
	$(SPIKE) $(SPIKE_FLAGS) $(IMAGE)

.PHONY: run
