"""Serial interface to the RIS controller.

The stock controller of the open-source surface accepts one configuration per
serial command. The exact command framing must be taken from the RIS firmware
documentation; it is kept configurable here (``prefix``/``suffix``) and must be
verified on the real hardware before use.
"""
from __future__ import annotations

import time

from .scheduler import N_ELEMENTS, to_bytes


class RISController:
    def __init__(self, port: str, baudrate: int = 115200,
                 prefix: bytes = b"", suffix: bytes = b"\n",
                 n_elements: int = N_ELEMENTS):
        import serial  # pyserial; imported lazily so tests run without hardware

        self.ser = serial.Serial(port, baudrate=baudrate, timeout=1)
        self.prefix, self.suffix, self.n_elements = prefix, suffix, n_elements

    def apply(self, cfg: int) -> None:
        # TODO: confirm framing (hex string vs raw bytes) against the firmware.
        payload = to_bytes(cfg, self.n_elements).hex().encode()
        self.ser.write(self.prefix + payload + self.suffix)
        self.ser.flush()

    def measure_update_limit(self, n: int = 500) -> float:
        """Send ``n`` configurations back-to-back and return achieved updates/s.

        This is the controller limit reported in the paper; repeat it several
        times and with an oscilloscope on an element if possible.
        """
        from .scheduler import next_configuration

        cfg = None
        start = time.perf_counter()
        for _ in range(n):
            cfg = next_configuration(cfg, self.n_elements)
            self.apply(cfg)
        return n / (time.perf_counter() - start)

    def close(self) -> None:
        self.ser.close()
