import sys
sys.path.insert(0, ".")
from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.transformer import TransformerBlockAutoRegressionTP
from software_model.utils import data_type_dict, Tensor
from software_model.layernorm import LayerNorm

# LayerNorm only knows "heuristic-TPU"; map the -new spelling (same tiling for layernorm)
_ln_cs = LayerNorm.compile_and_simulate
LayerNorm.compile_and_simulate = lambda self, pcb, mode: _ln_cs(
    self, pcb, "heuristic-TPU" if mode == "heuristic-TPU-new" else mode)

# ---- workload ----
D_MODEL, N_HEADS, BS, SEQ_LEN = 1536, 24, 1, 2048
# ---- memory-hierarchy overrides (not expressible in the JSON schema the parser reads) ----
L1_TO_L0_BYTES_PER_CYCLE_PER_CORE = 128                 # L1 -> L0, per core
HBM_CHANNELS = 32
HBM_BITS_PER_CYCLE_PER_CHANNEL = 1024 // HBM_CHANNELS   # 1024 bits/cycle total across all channels
HBM_TO_IO_DIE_CYCLES = 1          # hop latency HBM -> IO die
IO_DIE_TO_CHIPLET_CYCLES = 1      # hop latency IO die -> compute chiplet
WRITE_PACKET_BYTES = 128          # max payload per write packet (no metadata, no padding)
ONCHIP_WRITE_PACKET_CYCLES = 1    # per-packet latency for on-chip writes (array/L0 -> L1)
                                  # HBM write packets each pay the HBM hop latency above
L1_PARTITION = (0.25, 0.25, 0.5)   # (ifmap, weight, output) share of L1; None = shared-pool model
ZERO_OP_OVERHEAD = True            # False -> add the A100 per-kernel launch constants
COMPILE_MODE = "heuristic-TPU-new" # tiled simulator, L0 split (1,1,1) = 1 systolic array/core

# optional CLI overrides: --shared-pool, --hbm-per-channel, --no-hops, --no-write-packets
if "--shared-pool" in sys.argv:
    L1_PARTITION = None
if "--no-hops" in sys.argv:
    HBM_TO_IO_DIE_CYCLES = IO_DIE_TO_CHIPLET_CYCLES = 0
if "--no-write-packets" in sys.argv:
    WRITE_PACKET_BYTES = None
if "--hbm-per-channel" in sys.argv:          # 1024 bits/cycle PER channel (4096 B/cycle total)
    HBM_BITS_PER_CYCLE_PER_CHANNEL = 1024

specs = read_architecture_template("configs/wyvern_system.json")
dev_s = specs["device"]; chip = dev_s["compute_chiplet"]; core = chip["core"]; io = dev_s["io"]
n_chiplets = dev_s["compute_chiplet_count"]
freq = dev_s["frequency_Hz"]
total_cores = chip["core_count"] * n_chiplets

core.setdefault("SRAM_KB", core["L0_SRAM_KB"])                    # per-core SRAM = L0
io["global_buffer_MB"] = chip["L1_SRAM_KB"] * n_chiplets / 1024   # no L2 -> shared level is L1
io["global_buffer_bandwidth_per_cycle_byte"] = L1_TO_L0_BYTES_PER_CYCLE_PER_CORE * total_cores
io["memory_channel_active_count"] = HBM_CHANNELS                  # HBM bytes/s =
io["pin_count_per_channel"] = HBM_BITS_PER_CYCLE_PER_CHANNEL      #   channels * pins * bit/s / 8
io["bandwidth_per_pin_bit"] = freq                                # 1 bit/cycle per pin

system = template_to_system(specs)
system.device.compute_module.l2_partition = L1_PARTITION
# every HBM tile read/write pays the hop latency (not hidden by prefetch)
system.device.io_module.hop_latency_cycles = HBM_TO_IO_DIE_CYCLES + IO_DIE_TO_CHIPLET_CYCLES
# write packets: HBM writes pay the hop latency per packet; on-chip writes pay ONCHIP_WRITE_PACKET_CYCLES
system.device.io_module.write_packet_size_byte = WRITE_PACKET_BYTES
system.device.compute_module.l2_write_packet_size_byte = WRITE_PACKET_BYTES
system.device.compute_module.l2_write_packet_latency_cycles = ONCHIP_WRITE_PACKET_CYCLES
if ZERO_OP_OVERHEAD:
    ov = system.device.compute_module.overhead
    for k in ("matmul", "softmax", "layernorm", "gelu"):
        setattr(ov, k, 0)

print(f"cores={total_cores}  L0={core['SRAM_KB']}KB/core  shared(L1)={io['global_buffer_MB']}MB @ "
      f"{io['global_buffer_bandwidth_per_cycle_byte']} B/cyc  HBM={system.device.io_module.bandwidth/1e12:.3f} TB/s "
      f"({system.device.io_module.bandwidth/freq:.0f} B/cyc)  partition={L1_PARTITION}  "
      f"hbm_hops={system.device.io_module.hop_latency_cycles} cyc/transfer  write_pkt={WRITE_PACKET_BYTES} B")

dev = 1  # TP>1 hits an AllReduce shape bug in this branch
fp16 = data_type_dict["fp16"]
d_h = D_MODEL // N_HEADS; h = N_HEADS // dev
blk = TransformerBlockAutoRegressionTP(D_MODEL, N_HEADS, d_h, dev, fp16)
K_cache = Tensor([BS, h, d_h, SEQ_LEN], fp16)
V_cache = Tensor([BS, h, SEQ_LEN, d_h], fp16)
_ = blk(Tensor([BS, 1, D_MODEL], fp16), SEQ_LEN, K_cache, V_cache)

lat = blk.compile_and_simulate(system, COMPILE_MODE)
names = ("qkv q_mul_k a_mul_v h_matmul0 h_matmul1 h_matmul2 softmax "
         "layernorm0 layernorm1 gelu allreduce_mha allreduce_ffn").split()
for n, v in zip(names, blk.simluate_log.split(", ")):
    print(f"{n:12s} {float(v)*freq:12.0f} cycles")
print(f"per-block decode: {lat:.6e} s = {lat*freq:.0f} cycles")

print("chosen tiles (L2 M,N,K | L1 M,N,K):")
for n in ("Q_proj", "H_matmul0", "H_matmul1", "H_matmul2"):
    m = getattr(blk, n).best_mapping
    if m is not None:
        print(f"  {n:10s} {m.l2_tile_M},{m.l2_tile_N},{m.l2_tile_K} | {m.l1_tile_M},{m.l1_tile_N},{m.l1_tile_K}")
