#ifndef _MEMORY_IF_H
#define _MEMORY_IF_H

#include <stddef.h>
#include <stdint.h>

// 分配一块大小为size的内存，返回指向这块内存的指针
typedef void *(*mem_alloc_t)(void *self, size_t size);
// 释放一块内存，addr指向这块内存的指针
typedef void (*mem_free_t)(void *self, void *addr);
// 按指定字节数对齐分配，alignment 必须是 2 的幂
typedef void *(*mem_alloc_align_t)(void *self, size_t size, size_t alignment);

typedef struct memory_if {
    void             *self;
    mem_alloc_t       mem_alloc;
    mem_free_t        mem_free;
    mem_alloc_align_t mem_alloc_align;
} memory_if;

static inline void *MemoryIf_Alloc(memory_if *intf, size_t size) {
    return (intf && intf->mem_alloc) ? intf->mem_alloc(intf->self, size) : NULL;
}

static inline void MemoryIf_Free(memory_if *intf, void *addr) {
    if (intf && intf->mem_free && addr)
        intf->mem_free(intf->self, addr);
}

static inline void *MemoryIf_AllocAlign(memory_if *intf, size_t size, size_t alignment) {
    if (intf && intf->mem_alloc_align)
        return intf->mem_alloc_align(intf->self, size, alignment);
    return NULL;
}

#endif
