#include <stdint.h>

enum { BLOCKAGE = 1, POTHOLE = 2 };

static inline uint8_t decide(uint16_t total, uint16_t slow,
                             uint16_t adjacent, uint16_t adjacent_slow,
                             uint16_t merges, uint16_t threshold, uint8_t options) {
    uint8_t flags = (merges >= 3 || (options & 2u)) ? BLOCKAGE : 0;
    if (!(options & 1u) && threshold && slow <= total && adjacent_slow <= adjacent &&
        slow >= threshold && adjacent >= 3 &&
        5u * slow >= 3u * total && 4u * adjacent_slow <= adjacent)
        flags |= POTHOLE;
    return flags;
}
