"""USB request/reply protocol for the NUCLEO-F103RB alert policy."""
from functools import reduce
from operator import xor
import struct
import time


class Board:
    def __init__(self, port):
        import serial
        self.serial = serial.Serial(port, 9600, timeout=.05, write_timeout=.2, exclusive=True)
        self.sequence = 0

    def evaluate(self, total, slow, adjacent, adjacent_slow, merges, threshold, speed_breaker, visual_blockage=False):
        self.sequence = (self.sequence + 1) % 256
        counts = (total, slow, adjacent, adjacent_slow, merges, threshold)
        data = struct.pack('<BBB6H', 0xc7, self.sequence, int(speed_breaker) | (int(visual_blockage) << 1),
                           *(min(value, 65535) for value in counts))
        self.serial.reset_input_buffer()
        self.serial.write(data + bytes([reduce(xor, data)]))
        deadline = time.monotonic() + .4
        buffer = bytearray()
        while time.monotonic() < deadline:
            buffer.extend(self.serial.read(4))
            while len(buffer) >= 4:
                marker, sequence, flags, checksum = buffer[:4]
                if marker == 0xc8 and sequence == self.sequence and flags <= 3 and not reduce(xor, buffer[:4]):
                    return flags
                del buffer[0]
        raise RuntimeError('STM32 did not reply. Check the USB port and the Road Obstacle Detection firmware.')

    def close(self):
        self.serial.close()
