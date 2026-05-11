#include "shell.h"
#include "stdio.h"
#include "stdlib.h"

static void cmd_memtest(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: memtest <address>\n");
        return;
    }

    unsigned long addr = strtoul(argv[1], NULL, 0);
    printf("Running memory test at 0x%lX ...\n", addr);
}

static void cmd_echo(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++) {
        printf("%s ", argv[i]);
    }
    printf("\n");
}

int main(void)
{
    shell_init();

    shell_register_command("memtest", cmd_memtest, "Run memory test at <addr>");
    shell_register_command("echo",    cmd_echo,    "Echo back all arguments");

    shell_run();

    printf("Shell terminated.\n");
    return 0;
}