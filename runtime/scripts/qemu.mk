CC       = riscv64-unknown-elf-gcc
OBJDUMP  = riscv64-unknown-elf-objdump

CFLAGS  += -march=rv32imf -mabi=ilp32 -nostdlib -ffreestanding -O2 -Wall -Wextra -MMD
CFLAGS  += -I$(RUNTIME_DIR)/tinylibc/include

QEMU_DIR = $(RUNTIME_DIR)/src/qemu

SRCS    += $(QEMU_DIR)/start.S
SRCS    += $(QEMU_DIR)/init.c
SRCS    += $(wildcard $(RUNTIME_DIR)/tinylibc/src/*.c)

LDFLAGS += -T$(QEMU_DIR)/qemu.ld -nostdlib

QEMU = qemu-system-riscv32
QEMU_FLAGS = -machine virt -nographic 

run: $(IMAGE)
	@echo "  QEMU  $(notdir $(IMAGE))"
	@$(QEMU) $(QEMU_FLAGS) -bios $(IMAGE)

.PHONY: run
