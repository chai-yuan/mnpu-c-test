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
    bge t0, t1, 2f
1:
    sw zero, 0(t0)
    addi t0, t0, 4
    blt t0, t1, 1b
2:
    ret

_exit:
    wfi
    j _exit
