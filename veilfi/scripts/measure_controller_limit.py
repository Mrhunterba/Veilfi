"""Measure how many RIS configurations per second the controller accepts.

Usage:
    python scripts/measure_controller_limit.py --port /dev/ttyUSB0 --repeats 5
"""
import argparse
import statistics

from veilfi.shield.ris_serial import RISController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    ris = RISController(args.port, args.baud)
    try:
        rates = [ris.measure_update_limit(args.n) for _ in range(args.repeats)]
    finally:
        ris.close()
    print(f"updates/s: mean={statistics.mean(rates):.1f} "
          f"min={min(rates):.1f} max={max(rates):.1f}")


if __name__ == "__main__":
    main()
