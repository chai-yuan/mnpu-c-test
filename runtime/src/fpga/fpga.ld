OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
    ram (rwx) : ORIGIN = 0x80000000, LENGTH = 8K
}

SECTIONS {
    .text : {
        *(.text.entry)
        *(.text*)
    } > ram

    .rodata : {
        *(.rodata*)
    } > ram

    .data : {
        *(.data*)
    } > ram

    .bss : {
        _bss_start = .;
        *(.bss*)
        *(COMMON)
        _bss_end = .;
    } > ram

    _stack_top = ORIGIN(ram) + LENGTH(ram);
}
