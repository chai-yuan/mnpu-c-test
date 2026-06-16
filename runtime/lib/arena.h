#ifndef ARENA_H
#define ARENA_H

#include "interface/memory_if.h"
#include <stddef.h>

typedef struct Arena *ArenaHandle;

ArenaHandle Arena_Create(void *buffer, size_t size);
void *Arena_Alloc(ArenaHandle self, size_t size);
void  Arena_Reset(ArenaHandle self);

memory_if Arena_GetMemoryIf(ArenaHandle self);

#endif
