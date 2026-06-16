#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>

typedef struct Arena *ArenaHandle;

ArenaHandle Arena_Create(void *buffer, size_t size);
void       *Arena_Alloc(ArenaHandle self, size_t size);
void        Arena_Reset(ArenaHandle self);

#endif
