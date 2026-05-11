#include "shell.h"
#include "stdio.h"
#include "string.h"
#include <stddef.h>

static shell_command_t command_table[SHELL_MAX_COMMANDS];
static int command_count = 0;

static int shell_exit_request = 0;

void shell_init(void)
{
    command_count = 0;
    shell_exit_request = 0;
    shell_register_command("help", shell_cmd_help, "Show available commands");
    shell_register_command("exit", shell_cmd_exit, "Exit shell (also Ctrl+D)");
}

int shell_register_command(const char *name,
                           shell_cmd_handler_t handler,
                           const char *help)
{
    if (command_count >= SHELL_MAX_COMMANDS) {
        return -1;
    }

    for (int i = 0; i < command_count; i++) {
        if (strcmp(command_table[i].name, name) == 0) {
            command_table[i].handler = handler;
            command_table[i].help = help;
            return 0;
        }
    }

    command_table[command_count].name = name;
    command_table[command_count].handler = handler;
    command_table[command_count].help = help;
    command_count++;
    return 0;
}

static int shell_parse_line(char *line, int *argc, char *argv[])
{
    int count = 0;
    int in_token = 0;

    while (*line != '\0' && count < SHELL_MAX_ARGS) {
        if (*line == ' ' || *line == '\t') {
            if (in_token) {
                *line = '\0';
                in_token = 0;
            }
        } else {
            if (!in_token) {
                argv[count++] = line;
                in_token = 1;
            }
        }
        line++;
    }
    *argc = count;
    return count;
}

void shell_run(void)
{
    char input_line[SHELL_LINE_MAX];
    char *argv[SHELL_MAX_ARGS];
    int argc;

    printf("\nMiniShell (type 'help' for commands)\n");
    while (!shell_exit_request) {
        printf("> ");
        fflush(stdout);

        if (fgets(input_line, sizeof(input_line), stdin) == NULL) {
            printf("\n");
            shell_exit_request = 1;
            continue;
        }

        size_t len = strlen(input_line);
        if (len > 0 && input_line[len - 1] == '\n') {
            input_line[len - 1] = '\0';
        }

        shell_parse_line(input_line, &argc, argv);

        if (argc == 0) {
            continue;
        }

        int cmd_found = 0;
        for (int i = 0; i < command_count; i++) {
            if (strcmp(argv[0], command_table[i].name) == 0) {
                command_table[i].handler(argc, argv);
                cmd_found = 1;
                break;
            }
        }

        if (!cmd_found) {
            printf("Unknown command: '%s'. Type 'help' for list.\n", argv[0]);
        }
    }
}

void shell_cmd_help(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    printf("Available commands:\n");
    for (int i = 0; i < command_count; i++) {
        if (command_table[i].help) {
            printf("  %-12s - %s\n", command_table[i].name, command_table[i].help);
        } else {
            printf("  %s\n", command_table[i].name);
        }
    }
}

void shell_cmd_exit(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    shell_exit_request = 1;
}