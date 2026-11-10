# Validation: Wyvern decode cycle counts

This file records the LLMCompass code changes made to get decode-phase cycle counts for the
Wyvern hardware (`configs/wyvern_system.json`), and how to reproduce the results.

- Base commit: `f3cbf61` (branch `feature/dep_graph`)
- Workload: one GPT-style decode block (`TransformerBlockAutoRegressionTP`), d_model = 1536, 24 heads
  (d_head = 64), seq_len = 2048, batch size 1, fp16
- Hardware: Wyvern, simulated as **one device** (no tensor parallelism; see [Known limitations](#known-limitations))

## 1. Code changes

### 1.1 `software_model/matmul.py`: `heuristic-TPU` L2 tiling fix

**Problem.** The `heuristic-TPU` and `heuristic-TPU-new` modes always used the whole matrix as the
L2 tile (`l2_tile_M/N/K = M/N/K`). If the matrix doesn't fit in half the global buffer, `simulate()`
fails its capacity assertion. On Wyvern (4 MB shared buffer) a 1536×1536 fp16 weight is 4.7 MB,
so every decode matmul failed. The same thing happens on GA100 for large matmuls (for example 12288×12288).

**Fix.**
- New static method `Matmul._heuristic_tpu_l2_tiles(M, N, K, l2_words, l2_partition=None)`:
  - If the whole matmul fits, it returns `[(M, N, K)]`, the original behavior. **Results for matmuls
    that already fit are unchanged.**
  - Otherwise it returns the *maximal* tiles that fit: each dim is the full size or a power of two
    below it (down to 1, so M = 1 stays legal), and doubling any dim would no longer fit.
  - It asserts with a clear message if no tile fits.
- Both the `heuristic-TPU` and `heuristic-TPU-new` branches of `Matmul.compile_and_simulate` loop over
  these L2 tiles. The existing L1-tile search runs unchanged inside the loop.

### 1.2 `hardware_model/compute_module.py` + `software_model/matmul.py`: optional partitioned L1 (Wyvern)

**Model.** Wyvern's L1 (the simulator's "L2" / global buffer) is split into three fixed chunks:

| Chunk | Share of L1 | Holds / path |
|---|---|---|
| ifmap | 0.25 | input feature maps, streamed L1 → L0 |
| weight | 0.25 | weights, streamed L1 → systolic array directly (never stored in L0) |
| output | 0.5 | outputs and partial sums |

L0 (per-core SRAM) holds **only** ifmap tiles. All traffic (ifmap, weight, output and partial-sum
reads/writes) shares the single L1 port (128 B/cycle). Each chunk is double-buffered internally, so a
tile may use half its chunk.

**Changes.** The model is opt-in. With the default, `l2_partition=None`, behavior is identical to
before.
- `ComputeModule.__init__` takes a new keyword `l2_partition=None`. It is a tuple
  `(ifmap, weight, output)` of fractions of `l2_size`.
- New static method `Matmul._l2_tile_fits(m, n, k, l2_words, l2_partition)`:
  - with no partition, it checks the shared-pool rule `m·n + n·k + m·k ≤ l2_words`;
  - with a partition, it checks `m·k ≤ f_in·l2_words`, `k·n ≤ f_w·l2_words` and `m·n ≤ f_out·l2_words`.
  - It is used by `_heuristic_tpu_l2_tiles` and by the L2 capacity assertion in `Matmul.simulate()`.
- In the `heuristic-TPU` and `heuristic-TPU-new` L1-tile search, when partitioned:
  - `l1_tile_K_max` is limited only by the ifmap tile: `(SRAM_words / 2) // l1_tile_M`;
  - `l2_tile_N` is added as an `l1_tile_N` candidate, because weight tiles are no longer limited by L0.
- In `L2TileSimulator.simulate_l2_tile_compute_cycle_count`, when partitioned, weight tiles and
  output/partial-sum tiles are **not reused** between L1 batches. Every weight read, partial-sum read
  and output write is charged to the L1 port. Ifmap reuse is unchanged.
- In `L1TileSimulator.simulate_l1_tile_compute_cycle_count`, when partitioned, the L0 capacity
  assertion counts only the ifmap tile (`M·K`).

### 1.3 HBM → chiplet hop latency (`hardware_model/io_module.py`, `software_model/{matmul,softmax,layernorm,gelu}.py`)

**Problem.** HBM traffic was costed as bytes ÷ bandwidth only. `IOModule.latency` (1 µs) is set, but
every place that would use it is commented out. The interconnect `latency_second` only affects
device-to-device all-reduce. The `d2d`/`c2c` fields are read only by the cost/area model.

**Model.** Data from HBM crosses two hops before it reaches the compute chiplet:
HBM → IO die (1 cycle) and IO die → chiplet (1 cycle). **Every HBM transfer pays the full
2 cycles** (pessimistic: prefetching doesn't hide them).

**Changes.** The model is opt-in. With the default, `hop_latency_cycles=0`, behavior is identical to
before.
- `IOModule.__init__` takes a new keyword `hop_latency_cycles=0`.
- New method `IOModule.transfer_cycle_count(size_byte, clock_freq)` returns
  `hop_latency_cycles + ceil(size_byte / (bandwidth / clock_freq))`.
- The `L2TileSimulator.simulate_l2_tile_io_cycle_count` functions in `matmul.py`, `softmax.py` and
  `layernorm.py` now call `transfer_cycle_count`. So every HBM read or write of an L2 tile (ifmap, weight,
  partial sum and output) pays the hop latency.
- `GeLU.compile_and_simulate` adds `2 × hop_latency_cycles`, for its one HBM read and one HBM write.
- **Not covered:** the `heuristic-GPU` GEMV shortcut for M = 1 and the `roofline_model` paths stay
  bandwidth-only.

### 1.4 Write packets (`hardware_model/{io_module,compute_module}.py`, `software_model/{matmul,softmax,layernorm,gelu}.py`)

**Problem.** Writes were costed as raw bytes ÷ bandwidth, with no packets. Packet overhead
(`flit_size`, `header_size`, `max_payload_size`) was only used by device-to-device all-reduce.

**Model.** Every write (outputs and partial sums) is split into packets of at most **128 B**. There is no
metadata, and the last packet carries only the remaining bytes (no padding). So the bandwidth time is
unchanged, and the extra cost is a fixed latency per packet:

| Write | Per-packet latency | Cost of one write of `B` bytes |
|---|---|---|
| chiplet → HBM | the 2-cycle HBM hop (§1.3) | `ceil(B/128) × 2 + ceil(B ÷ HBM B/cycle)` |
| on-chip: systolic array / L0 → L1 | 1 cycle | `ceil(B/128) × 1 + ceil(B ÷ 128 B/cycle)` |

Packets are sent one after another, and each one pays its latency (no overlap between packets). Reads
are unchanged. An HBM read still pays the hop latency once per transfer.

**Changes.** The model is opt-in. With the defaults (`None` / `0`), behavior is identical to before.
- `IOModule.__init__` takes a new keyword `write_packet_size_byte=None`. New methods:
  - `write_packet_count(size_byte)`: `ceil(size/packet)`, or 1 when packets are off.
  - `write_cycle_count(size_byte, clock_freq)`: `packets × hop_latency_cycles + ceil(size ÷ bandwidth)`.
- `ComputeModule.__init__` takes new keywords `l2_write_packet_size_byte=None` and
  `l2_write_packet_latency_cycles=0`. New methods:
  - `l2_write_packet_count(size_byte)`: the number of packets in one on-chip write.
  - `l2_write_packet_cycle_count(packet_count, parallel_ports=1)`:
    `ceil(packets ÷ ports) × latency`. Multi-core writes are spread over `core_count` per-core ports.
- HBM write sites now use `IOModule.write_cycle_count`:
  - `matmul.py`: a new `L2TileSimulator.M_N_write_cycle_count` is used for the previous-tile write-back
    and the final-tile write in `Matmul.simulate()`. Partial-sum *reads* still use `M_N_io_cycle_count`.
  - `softmax.py`, `layernorm.py`: the L2 tile `write_cycle_count`.
  - `gelu.py`: the HBM write pays the hop latency per packet, and the read pays it once.
- On-chip write sites add `l2_write_packet_cycle_count`:
  - `matmul.py`: the L1-batch output writes and the last-batch write in
    `L2TileSimulator.simulate_l2_tile_compute_cycle_count`, counting packets per output tile. Also the
    write half of `K_reduction_cycle_count`.
  - `softmax.py`, `layernorm.py`: the L1 tile `write_cycle_count` and the write half of
    `reduction_cycle_count`.
  - `gelu.py`: the output write back to L1.

### 1.5 Side effects in the working tree

- `systolic_array_model/look_up_table_128_128.csv` (tracked): scalesim appended 23 rows for
  systolic-array tile shapes that weren't in the table. They are produced automatically the first time a
  new shape is simulated. Commit them to make later runs faster, or revert them; results are the same either way.
- `systolic_array_model/temp/` (gitignored): the working directory scalesim needs. The code doesn't
  create it, so it must exist before the first run with new tile shapes.

### 1.6 Not changed (handled in the run script instead)

These are worked around in the script in §2.3 and the repo code is left as-is:

| Issue | Where | Workaround in script |
|---|---|---|
| `template_to_system` requires `core.SRAM_KB`; Wyvern uses `L0_SRAM_KB` | `design_space_exploration/dse.py` | set `SRAM_KB = L0_SRAM_KB` |
| Wyvern sets `global_buffer_MB: 0` (softmax divides by zero) | config / `software_model/softmax.py` | global buffer = `L1_SRAM_KB × compute_chiplet_count` |
| The JSON can't express the L1→L0 bandwidth or HBM in bits/cycle | config schema | overridden in the script (§2.2) |
| `heuristic-TPU` hard-codes an L0 split over 2 systolic arrays/core; Wyvern has 1 | `matmul.py`, L1TileSimulator assertion | use `heuristic-TPU-new`, which has a 1-array split |
| `LayerNorm` only accepts the string `"heuristic-TPU"` | `software_model/layernorm.py` | the script maps `-new` → `heuristic-TPU` for LayerNorm |
| A100 per-kernel launch overheads (21–45 µs per operator) are added to every operator | `overhead_dict["A100"]` | set to 0 (`ZERO_OP_OVERHEAD = True`) |

## 2. Reproducing the results

Run every command from the repository root. The simulator loads
`./systolic_array_model/look_up_table_*.csv` relative to the current directory.

### 2.1 One-time setup

```bash
cd LLMCompass
mkdir -p systolic_array_model/temp   # scalesim scratch dir (gitignored)
mkdir -p validation
```

### 2.2 Hardware configuration used

`configs/wyvern_system.json` as of this validation (the relevant fields):

| Field | Value |
|---|---|
| `frequency_Hz` | 1e9 |
| `compute_chiplet_count` | 1 |
| `compute_chiplet.core_count` | 1 |
| systolic array | 128 × 128, fp16, `sublane_count: 1` |
| vector unit | width 64, 4 flop/cycle |
| `core.L0_SRAM_KB` | 2048 |
| `L1_SRAM_KB` / `L2_SRAM_KB` | 4096 / 0 |

The script then applies these overrides:

| Parameter | Value | How it's applied |
|---|---|---|
| L0 (per-core SRAM) | 2 MB | `SRAM_KB = L0_SRAM_KB` |
| Shared level (L1 used as the "global buffer") | 4 MB | `global_buffer_MB = L1_SRAM_KB × chiplets / 1024` |
| L1 → L0 bandwidth | 128 B/cycle per core | `global_buffer_bandwidth_per_cycle_byte = 128 × cores` |
| HBM bandwidth | 1024 bits/cycle **total** across 32 channels = 128 B/cycle (128 GB/s) | 32 channels × 32 pins × 1 Gbit/s |
| L1 partition | (0.25, 0.25, 0.5) | `compute_module.l2_partition` |
| HBM hop latency | HBM → IO die 1 cycle + IO die → chiplet 1 cycle = 2 cycles per transfer | `io_module.hop_latency_cycles` |
| Write packets | max 128 B payload, no metadata, no padding; HBM write packets pay 2 cycles each, on-chip write packets pay 1 cycle each | `io_module.write_packet_size_byte`, `compute_module.l2_write_packet_size_byte` / `l2_write_packet_latency_cycles` |

The `io` block in the JSON (4 channels × 1024 pins × 3.2 Gbit/s) is **ignored**; the script overrides it.

### 2.3 Decode script

Save as `validation/wyvern_decode_cycles.py`:

```python
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
```

Run:

```bash
python validation/wyvern_decode_cycles.py                                     # (A) partitioned L1
python validation/wyvern_decode_cycles.py --shared-pool                       # (B) shared-pool L1
python validation/wyvern_decode_cycles.py --shared-pool --hbm-per-channel     # (C) shared pool, 4096 B/cycle HBM
```

Add flags to any of these to turn individual effects off:

| Flag | Effect |
|---|---|
| `--no-write-packets` | writes are not packetized (bytes ÷ bandwidth, hop latency once per HBM write) |
| `--no-hops` | no HBM hop latency (with packets on, on-chip write packets still pay 1 cycle each) |

`--no-write-packets --no-hops` gives the pure bandwidth model.

Cycles = simulated latency (s) × `frequency_Hz`. Each run takes about 10–20 s once the lookup
table has the needed shapes.

### 2.4 Expected results

Cycles for one decode block, with operator overhead set to 0 and `heuristic-TPU-new`.

**Main results: 2-cycle HBM hop latency plus 128 B write packets** (default flags):

| Operator | (A) partitioned L1 | (B) shared pool | (C) shared pool, HBM 4096 B/cycle |
|---|---:|---:|---:|
| qkv (3 × Q_proj) | 145,413 | 155,799 | 119,997 |
| q_mul_k | 71,854 | 65,367 | 56,004 |
| a_mul_v | 106,678 | 85,848 | 77,766 |
| h_matmul0 | 48,471 | 51,933 | 39,999 |
| h_matmul1 | 174,520 | 169,434 | 160,925 |
| h_matmul2 | 181,716 | 170,616 | 158,682 |
| softmax | 27,978 | 27,978 | 26,490 |
| layernorm0 / layernorm1 | 272 / 272 | 272 / 272 | 226 / 226 |
| gelu | 1,080 | 1,080 | 1,080 |
| allreduce ×2 | 0 | 0 | 0 |
| **Total per block** | **758,254** | **728,599** | **641,395** |

With `--no-write-packets` (hop latency only, once per HBM transfer):

| Operator | (A) partitioned L1 | (B) shared pool | (C) shared pool, HBM 4096 B/cycle |
|---|---:|---:|---:|
| qkv (3 × Q_proj) | 144,651 | 155,229 | 119,427 |
| q_mul_k | 71,756 | 64,907 | 55,883 |
| a_mul_v | 106,652 | 85,834 | 77,752 |
| h_matmul0 | 48,217 | 51,743 | 39,809 |
| h_matmul1 | 172,368 | 167,230 | 158,733 |
| h_matmul2 | 180,652 | 169,994 | 158,060 |
| softmax | 25,617 | 25,617 | 24,129 |
| layernorm0 / layernorm1 | 202 / 202 | 202 / 202 | 156 / 156 |
| gelu | 1,080 | 1,080 | 1,080 |
| allreduce ×2 | 0 | 0 | 0 |
| **Total per block** | **751,397** | **722,038** | **635,185** |

With `--no-write-packets --no-hops` (bandwidth only):

| Operator | (A) partitioned L1 | (B) shared pool | (C) shared pool, HBM 4096 B/cycle |
|---|---:|---:|---:|
| qkv (3 × Q_proj) | 144,585 | 155,211 | 119,409 |
| q_mul_k | 71,718 | 64,901 | 55,869 |
| a_mul_v | 106,646 | 85,828 | 77,746 |
| h_matmul0 | 48,195 | 51,737 | 39,803 |
| h_matmul1 | 172,292 | 167,190 | 158,693 |
| h_matmul2 | 180,576 | 169,988 | 158,054 |
| softmax | 25,613 | 25,613 | 24,125 |
| layernorm0 / layernorm1 | 198 / 198 | 198 / 198 | 152 / 152 |
| gelu | 1,080 | 1,080 | 1,080 |
| allreduce ×2 | 0 | 0 | 0 |
| **Total per block** | **751,101** | **721,944** | **635,083** |

How much each effect adds to the total:

| Step | (A) | (B) | (C) |
|---|---:|---:|---:|
| bandwidth only → + hop latency | +296 | +94 | +102 |
| hop latency → + write packets | +6,857 (+0.9%) | +6,561 (+0.9%) | +6,210 (+1.0%) |

- **Hop latency is small.** Each HBM transfer pays it only once, and each transfer is thousands of
  cycles of bandwidth time. (A) pays the most because its smaller L2 tiles mean more transfers.
- **Write packets matter more.** A write of `B` bytes now pays `ceil(B/128)` latencies instead of one.
  Softmax (24 × 2049 fp16 outputs) and the FFN matmuls, which have the largest outputs, account for most
  of the increase.
- **gelu is unchanged** in every case. It is compute-bound, and its `max(compute, io)` hides the extra
  cycles.

Chosen tiles, shown as L2 tile (M,N,K) | L1 tile (M,N,K):

| Matmul | (A) partitioned | (B) shared pool | (C) shared pool, HBM 4096 B/cycle |
|---|---|---|---|
| Q_proj, H_matmul0, H_matmul2 | 1,512,512 \| 1,128,512 | 1,1536,512 \| 1,128,512 | 1,1536,512 \| 1,128,512 |
| H_matmul1 | 1,1024,256 \| 1,128,256 | 1,2048,256 \| 1,128,256 | 1,512,1536 \| 1,128,1024 with write packets; 1,2048,256 \| 1,128,256 without |

For the full model, multiply the per-block total by the number of layers. The lm_head matmul is not
included.

### 2.5 Sanity checks

- **Bandwidth floor.** Every weight matmul must take at least weight bytes ÷ 128 B/cycle. For example,
  h_matmul0 ≥ 1536·1536·2 / 128 = 36,864 and h_matmul1 ≥ 1536·6144·2 / 128 = 147,456. All results above
  satisfy this.
- **Why (A) differs from (B).** Under (A) the 512 KB weight chunk forces smaller L2 tiles, so the first
  HBM fill (not overlapped) is shorter. That makes qkv and h_matmul0 faster. `BatchedMatmul` folds the
  24 heads of a_mul_v into one 1×49,176×64 matmul. Its L2 tile K halves (8192 → 4096), and partial
  sums now cross the L1 port on every K step, so a_mul_v gets slower.
- **Hop latency accounting.** Each operator's increase from hops is 2 cycles × the number of HBM
  transfers that aren't hidden by compute. For example, h_matmul0 in (A) rises by 22 cycles, which
  is 11 transfers on the critical path for its 3×3 grid of 512×512 weight tiles.
- **Write packet accounting (layernorm, worked by hand).** A layernorm output row is 1536 fp16 =
  3,072 B = 24 packets. The HBM write goes from 1 × 2 cycles to 24 × 2 cycles (+46). The on-chip write
  adds 24 × 1 cycle (+24). Together that's +70, which matches layernorm going from 202 to 272 in (A) and (B).
- **Write packet accounting (softmax, HBM part).** The softmax output is 24 × 2049 fp16 = 98,352 B =
  769 packets. The HBM write goes from 2 to 769 × 2 cycles (+1,536). The rest of its +2,361 comes from
  on-chip write packets (the L1 tile writes and reduction writes).

### 2.6 Regression check (existing configs unchanged)

This checks that the changes don't alter results for matmuls that already fit the global buffer.
Save as `validation/regress_ga100.py`:

```python
import sys; sys.path.insert(0, ".")
from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
fp16 = data_type_dict["fp16"]
system = template_to_system(read_architecture_template("configs/GA100.json"))
for mode in ("heuristic-TPU", "heuristic-TPU-new"):
    for M, N, K in [(1, 2048, 2048), (256, 2048, 2048)]:
        mm = Matmul(fp16); _ = mm(Tensor([M, K], fp16), Tensor([K, N], fp16))
        print(mode, (M, N, K), mm.compile_and_simulate(system.device, mode))
```

```bash
python validation/regress_ga100.py
```

Expected output: it is identical before the changes (commit `f3cbf61`) and after them.

```
heuristic-TPU (1, 2048, 2048) 5.977304964539007e-06
heuristic-TPU (256, 2048, 2048) 1.7208510638297873e-05
heuristic-TPU-new (1, 2048, 2048) 6.688652482269503e-06
heuristic-TPU-new (256, 2048, 2048) 2.8947517730496454e-05
```

To compare against the original code yourself, run
`git stash push software_model/matmul.py software_model/softmax.py software_model/layernorm.py software_model/gelu.py hardware_model/compute_module.py hardware_model/io_module.py`, run the script, then `git stash pop`.

## Known limitations

- **Single device only.** With `device_count > 1`, `AllReduceMultiPCB.__call__` expects a 4-D tensor
  but gets a 3-D one (`software_model/communication_primitives.py:35`). So tensor parallelism across
  Wyvern's 4 devices isn't simulated, and the all-reduce cost is 0.
- **Partitioned L1 support is limited to the heuristic-TPU modes.** It is only implemented in
  `heuristic-TPU` and `heuristic-TPU-new`. `exhaustive` and `heuristic-our-throughput` ignore the
  partition when choosing tiles and can trip the capacity assertion if `l2_partition` is set.
- **`heuristic-GPU` doesn't use the tiled model at M = 1.** Matmuls with M = 1 skip the tiled simulator
  and use a GEMV roofline (max of HBM time and vector-unit time). They ignore the systolic array, the
  L1→L0 link and the partition. That's why these results use `heuristic-TPU-new`.
- **Softmax, layernorm and gelu** use their own simple models. They don't model the L1 partition or the
  systolic array.
- **The decode block's `compile_and_simulate` is a serial sum.** It adds up operator latencies (Q_proj
  ×3 for Q/K/V, layernorm ×2) and ignores the core assignments in the dependency graph and any overlap
  between operators.
- **HBM hop latency is a fixed per-transfer charge.** It isn't overlapped with other transfers
  (pessimistic), it doesn't scale with distance (no c2c ring hops, since the config has one chiplet), and
  it isn't applied to the `heuristic-GPU` M = 1 GEMV shortcut or the roofline models.
- **Write packets are serialized.** Each packet pays its full latency, with no overlap between
  packets. On multi-core configs, on-chip packets are spread evenly over the per-core ports
  (`ceil(packets ÷ core_count)`), which is exact only for 1 core or equal-size tiles. The roofline,
  `heuristic-GPU` GEMV and all-reduce paths don't use these write packets.
- **`topology: "MESH"` is treated as `RING`** by `template_to_system`. This doesn't matter for
  single-device runs.
