#ifndef _TIME_H
#define _TIME_H

#ifndef _CLOCK_T_DEFINED
#define _CLOCK_T_DEFINED
typedef unsigned long clock_t;
#endif

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC 1000000UL
#endif

clock_t clock(void);

#endif /* _TIME_H */
