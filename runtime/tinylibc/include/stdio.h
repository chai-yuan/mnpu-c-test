#ifndef _STDIO_H
#define _STDIO_H

typedef void FILE;

#define stdin ((FILE *)0)
#define stdout ((FILE *)1)
#define stderr ((FILE *)2)

int   printf(const char *fmt, ...);
int   fprintf(void *stream, const char *fmt, ...);
int   getchar(void);
char *fgets(char *s, int size, void *stream);
int   fflush(void *stream);

#endif
