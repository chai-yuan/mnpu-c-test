CC       = riscv64-unknown-elf-gcc
CFLAGS  += -march=rv32imf -mabi=ilp32 -nostdlib -ffreestanding -O2 -Wall -Wextra -MMD
CFLAGS  += -I$(RUNTIME_DIR)/tinylibc/include

SRCS    += $(RUNTIME_DIR)/qemu/start.S
SRCS    += $(RUNTIME_DIR)/qemu/init.c
SRCS    += $(wildcard $(RUNTIME_DIR)/tinylibc/src/*.c)

LDFLAGS += -T$(RUNTIME_DIR)/qemu/qemu.ld -nostdlib

QEMU = qemu-system-riscv32
QEMU_FLAGS = -M virt -nographic -bios

run: $(IMAGE)
	@echo "  QEMU  $(notdir $(IMAGE))"
	@$(QEMU) $(QEMU_FLAGS) $(IMAGE)

.PHONY: run
