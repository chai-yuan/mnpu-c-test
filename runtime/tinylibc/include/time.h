#ifndef _TIME_H
#define _TIME_H

#include <stdint.h>

#ifndef _CLOCK_T_DEFINED
#define _CLOCK_T_DEFINED
typedef uint64_t clock_t;
#endif

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC 1000000UL
#endif

clock_t clock(void);

#endif /* _TIME_H */
