"""Read 802.11 headers from a monitor-mode pcap into frame records.

Capture headers only, e.g.:
    sudo tcpdump -i wlan1mon -s 128 -w session.pcap
(-s 128 truncates frames so payloads are not stored.)
"""
from __future__ import annotations

FTYPE = {0: "mgmt", 1: "ctrl", 2: "data"}
SUBTYPE = {(0, 8): "beacon", (0, 4): "probe_req", (0, 5): "probe_resp",
           (1, 13): "ack", (1, 11): "rts", (1, 12): "cts",
           (2, 0): "data", (2, 4): "null", (2, 8): "qos_data", (2, 12): "qos_null"}


def read_frames(pcap_path: str, members: set[str]):
    from scapy.all import Dot11, PcapReader  # lazy import

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if not pkt.haslayer(Dot11):
                continue
            d = pkt[Dot11]
            src = (d.addr2 or "").lower()
            yield {
                "t": float(pkt.time),
                "src": src,
                "dst": (d.addr1 or "").lower(),
                "ftype": FTYPE.get(d.type, "other"),
                "subtype": SUBTYPE.get((d.type, d.subtype), f"{d.type}:{d.subtype}"),
                # ACK/CTS frames carry no transmitter address
                "associated": src in members if src else True,
            }
