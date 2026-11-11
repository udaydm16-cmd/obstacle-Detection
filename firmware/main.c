#include "policy.h"

#define REG(address) (*(volatile uint32_t *)(address))
#define UART 0x40004400u
#define GPIOA 0x40010800u

extern uint32_t _stack;
void reset(void);
static void halt(void) { for (;;) {} }

__attribute__((section(".vectors"), used))
const uintptr_t vectors[16] = {
    (uintptr_t)&_stack, (uintptr_t)reset, (uintptr_t)halt, (uintptr_t)halt,
    (uintptr_t)halt, (uintptr_t)halt, (uintptr_t)halt, 0, 0, 0, 0,
    (uintptr_t)halt, (uintptr_t)halt, 0, (uintptr_t)halt, (uintptr_t)halt
};

static uint16_t word(const uint8_t *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static void send(uint8_t byte) {
    while (!(REG(UART) & (1u << 7))) {}
    REG(UART + 4) = byte;
}

void reset(void) {
    REG(0x40021018u) |= 1u << 2;
    REG(0x4002101cu) |= 1u << 17;
    REG(GPIOA) = (REG(GPIOA) & ~((15u << 8) | (15u << 12) | (15u << 20)))
               | (10u << 8) | (4u << 12) | (2u << 20);
    REG(UART + 8) = 833u;
    REG(UART + 12) = (1u << 13) | (1u << 3) | (1u << 2);

    uint8_t packet[16];
    unsigned used = 0;
    for (;;) {
        uint32_t status = REG(UART);
        if (!(status & 0x2fu)) continue;
        uint8_t byte = (uint8_t)REG(UART + 4);
        if (status & 15u) { used = 0; continue; }
        if (!used && byte != 0xc7) continue;
        packet[used++] = byte;
        if (used != sizeof packet) continue;
        used = 0;
        uint8_t check = 0;
        for (unsigned i = 0; i < sizeof packet; ++i) check ^= packet[i];
        if (check) continue;
        uint8_t flags = decide(word(packet+3), word(packet+5), word(packet+7),
                               word(packet+9), word(packet+11), word(packet+13), packet[2]);
        if (flags) REG(GPIOA + 16) = 1u << 5;
        send(0xc8);
        send(packet[1]);
        send(flags);
        send(0xc8 ^ packet[1] ^ flags);
    }
}
