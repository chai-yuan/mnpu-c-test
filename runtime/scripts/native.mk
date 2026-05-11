CC       = gcc
OBJDUMP  = objdump

CFLAGS  += -O2 -Wall -Wextra -MMD
LDFLAGS +=

run: $(IMAGE)
	@echo "  RUN  $(notdir $(IMAGE))"
	@$(IMAGE)

.PHONY: run