CROSS ?= arm-none-eabi-
CFLAGS := -mcpu=cortex-m3 -mthumb -Os -Wall -Wextra -Werror -ffreestanding -ffunction-sections -fdata-sections

hazard.elf: main.c policy.h f103rb.ld
	$(CROSS)gcc $(CFLAGS) -nostdlib -Wl,--gc-sections -T f103rb.ld main.c -o $@
	$(CROSS)size $@
