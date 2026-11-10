from math import ceil


class IOModule:
    def __init__(
        self, bandwidth, latency, hop_latency_cycles=0, write_packet_size_byte=None
    ):
        self.bandwidth = bandwidth
        self.latency = latency
        # Fixed delay (cycles) every off-chip transfer pays on its way between
        # memory and the compute chiplet, e.g. HBM -> IO die + IO die -> chiplet.
        # Charged per transfer, not hidden by prefetch. 0 = original behavior.
        self.hop_latency_cycles = hop_latency_cycles
        # Max payload (bytes) of a write packet to memory. When set, a write is
        # split into ceil(size / packet) packets (no padding, no metadata) and
        # every packet pays hop_latency_cycles. None = original behavior.
        self.write_packet_size_byte = write_packet_size_byte

    def transfer_cycle_count(self, size_byte, clock_freq):
        """Cycles to move size_byte between memory and the chiplet."""
        return self.hop_latency_cycles + ceil(size_byte / (self.bandwidth / clock_freq))

    def write_packet_count(self, size_byte):
        """Number of write packets for size_byte (1 if writes aren't packetized)."""
        if not self.write_packet_size_byte:
            return 1
        return max(1, ceil(size_byte / self.write_packet_size_byte))

    def write_cycle_count(self, size_byte, clock_freq):
        """Cycles to write size_byte from the chiplet to memory."""
        return self.write_packet_count(size_byte) * self.hop_latency_cycles + ceil(
            size_byte / (self.bandwidth / clock_freq)
        )


IO_module_dict = {
    "A100": IOModule(2039e9, 1e-6),
    "TPUv3": IOModule(float("inf"), 1e-6),
    "MI210": IOModule(1.6e12, 1e-6)
}
