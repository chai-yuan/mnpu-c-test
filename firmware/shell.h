#ifndef SHELL_H
#define SHELL_H

#define SHELL_MAX_COMMANDS 16
#define SHELL_LINE_MAX 128
#define SHELL_MAX_ARGS 8

typedef void (*shell_cmd_handler_t)(int argc, char *argv[]);

typedef struct {
    const char         *name;
    shell_cmd_handler_t handler;
    const char         *help;
} shell_command_t;

void shell_init(void);

int shell_register_command(const char *name, shell_cmd_handler_t handler, const char *help);

void shell_run(void);

void shell_cmd_help(int argc, char *argv[]);

void shell_cmd_exit(int argc, char *argv[]);

#endif /* SHELL_H */