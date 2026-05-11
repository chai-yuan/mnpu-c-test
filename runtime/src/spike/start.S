.section .text.entry
.globl _start
_start:
    la sp, _stack_top

    call _clear_bss

    # 开启 FPU
    li t0, 0x00002000
    csrs mstatus, t0

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
    la t0, tohost
    li t1, 0
    sw t1, 4(t0)
    li t1, 1
    sw t1, 0(t0)
_exit_loop:
    wfi
    j _exit_loop
