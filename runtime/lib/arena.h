#ifndef ARENA_H
#define ARENA_H

#include "interface/memory_if.h"
#include <stddef.h>

typedef struct Arena *Arena;

Arena Arena_Create(void *buffer, size_t size);
void *Arena_Alloc(Arena self, size_t size);
void *Arena_Free(Arena self, void *ptr);
void  Arena_Reset(Arena self);

memory_if Arena_GetMemoryIf(Arena self);

#endif
