.section .text.entry
.globl _start
_start:
    la sp, _stack_top

    call _clear_bss

    call _init
    j _exit

_clear_bss:
    la t0, _bss_start
    la t1, _bss_end
    bge t0, t1, _clear_bss_end
_clear_bss_loop:
    sw zero, 0(t0)
    addi t0, t0, 4
    blt t0, t1, _clear_bss_loop
_clear_bss_end: 
    ret

_exit:
    li t0, 0x100000
    li t1, 0x5555
    sw t1, 0(t0)
_exit_loop:
    wfi
    j _exit_loop
