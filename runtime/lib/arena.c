#include "arena.h"

#define ARENA_ALIGN 16

struct Arena {
    char  *start;
    char  *current;
    size_t capacity;
};

static size_t align_up(size_t size) { return (size + ARENA_ALIGN - 1) & ~((size_t)ARENA_ALIGN - 1); }

Arena Arena_Create(void *buffer, size_t size) {
    size_t header_size = align_up(sizeof(struct Arena));
    if (size < header_size)
        return NULL;

    struct Arena *arena = (struct Arena *)buffer;
    arena->start        = (char *)buffer + header_size;
    arena->current      = arena->start;
    arena->capacity     = size - header_size;
    return arena;
}

void *Arena_Alloc(Arena handle, size_t size) {
    struct Arena *arena   = (struct Arena *)handle;
    size_t        aligned = align_up(size);
    size_t        used    = (size_t)(arena->current - arena->start);
    if (used + aligned > arena->capacity)
        return NULL;
    void *ptr = arena->current;
    arena->current += aligned;
    return ptr;
}

void *Arena_Free(Arena self, void *ptr) {
    // NULL
}

void Arena_Reset(Arena handle) {
    struct Arena *arena = (struct Arena *)handle;
    arena->current      = arena->start;
}

memory_if Arena_GetMemoryIf(Arena self) {
    return (memory_if){
        .self            = self,
        .mem_alloc       = Arena_Alloc,
        .mem_free        = Arena_Free,
        .mem_alloc_align = NULL,
    };
}