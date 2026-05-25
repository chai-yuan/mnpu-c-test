#include "time.h"
#include "port.h"

clock_t clock(void) {
    clock_t cycles = 0;
    if (port_global.get_time)
        cycles = port_global.get_time();
    return cycles;
}
