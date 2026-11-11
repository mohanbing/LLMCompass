# Scratchpad (SRAM) area model backed by CACTI 7
# Build CACTI first:
#   git clone https://github.com/HewlettPackard/cacti.git third_party/cacti
#   cd third_party/cacti && make opt
#
# CACTI only models technology nodes down to 22nm. For smaller nodes we run CACTI
# at 22nm to get the array organization and area efficiency (cell area / total area),
# then rescale using the target node's published SRAM bitcell area from cost_model.py:
#   area = bits * target_bitcell_area / area_efficiency
# this uses the published bitcells, not the calibrated sram_bit_cell_density_* values,
# because CACTI models the array periphery itself (see the comment in cost_model.py)

import argparse
import functools
import os
import re
import subprocess
import tempfile

from cost_model.cost_model import find_published_sram_bitcell_area

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACTI_PATH = os.environ.get("CACTI_PATH", os.path.join(REPO_ROOT, "third_party", "cacti"))
CFG_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cacti", "scratchpad.cfg.template")

# technology nodes CACTI 7 models natively
CACTI_NATIVE_NODES_NM = [90, 65, 45, 32, 22]
CACTI_MIN_NODE_NM = 22

# CACTI rejects arrays smaller than this
CACTI_MIN_SIZE_BYTES = 64

# weights: delay, dynamic power, leakage power, cycle time, area
AREA_OPTIMAL_OBJECTIVE = "0:0:0:0:100"
NO_DEVIATION_LIMIT = "100000:100000:100000:100000:100000"


def _parse_float(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


@functools.lru_cache(maxsize=None)
def run_cacti(size_bytes, block_bytes=64, banks=1, rw_ports=1, read_ports=0, write_ports=0,
              tech_nm=CACTI_MIN_NODE_NM, cell_type="itrs-hp", peripheral_type="itrs-hp",
              objective=AREA_OPTIMAL_OBJECTIVE, deviate=NO_DEVIATION_LIMIT):
    """Run CACTI on a scratch RAM and return its area/timing/energy results at tech_nm."""
    cacti_bin = os.path.join(CACTI_PATH, "cacti")
    if not os.path.isfile(cacti_bin):
        raise FileNotFoundError(
            f"CACTI binary not found at {cacti_bin}. Build it with "
            f"`cd {CACTI_PATH} && make opt` or set CACTI_PATH.")
    if tech_nm not in CACTI_NATIVE_NODES_NM:
        raise ValueError(f"CACTI supports {CACTI_NATIVE_NODES_NM} nm, got {tech_nm}")

    size_bytes = max(int(size_bytes), CACTI_MIN_SIZE_BYTES)
    with open(CFG_TEMPLATE) as f:
        cfg = f.read().format(
            size_bytes=size_bytes,
            block_bytes=block_bytes,
            banks=banks,
            rw_ports=rw_ports,
            read_ports=read_ports,
            write_ports=write_ports,
            tech_um=tech_nm / 1000,
            cell_type=cell_type,
            peripheral_type=peripheral_type,
            bus_width_bits=block_bytes * 8,
            objective=objective,
            deviate=deviate,
        )

    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as cfg_file:
        cfg_file.write(cfg)
        cfg_path = cfg_file.name
    try:
        result = subprocess.run([cacti_bin, "-infile", cfg_path], cwd=CACTI_PATH,
                                capture_output=True, text=True)
    finally:
        os.remove(cfg_path)

    out = result.stdout + result.stderr
    area_mm2 = _parse_float(r"Data array: Area \(mm2\):\s*([\d.eE+-]+)", out)
    efficiency_pct = _parse_float(r"Area efficiency \(Memory cell area/Total area\) - ([\d.eE+-]+)", out)
    if area_mm2 is None or efficiency_pct is None:
        tail = "\n".join(out.strip().splitlines()[-10:])
        raise RuntimeError(f"CACTI did not produce a valid organization for {size_bytes} B:\n{tail}")

    bits = size_bytes * 8
    efficiency = efficiency_pct / 100
    return {
        "size_bytes": size_bytes,
        "tech_nm": tech_nm,
        "area_mm2": area_mm2,
        "area_efficiency": efficiency,
        "bitcell_area_mm2": area_mm2 * efficiency / bits,
        "access_time_ns": _parse_float(r"Access time \(ns\):\s*([\d.eE+-]+)", out),
        "cycle_time_ns": _parse_float(r"Cycle time \(ns\):\s*([\d.eE+-]+)", out),
        "read_energy_nJ": _parse_float(r"Total dynamic read energy per access \(nJ\):\s*([\d.eE+-]+)", out),
        "write_energy_nJ": _parse_float(r"Total dynamic write energy per access \(nJ\):\s*([\d.eE+-]+)", out),
        "leakage_mW": _parse_float(r"Total leakage power of a bank \(mW\):\s*([\d.eE+-]+)", out),
    }


def _node_nm(process_node):
    return int(re.match(r"\s*(\d+)", str(process_node)).group(1))


def scratchpad_area_mm2(size_bytes, process_node="7nm", bitcell_area_mm2=None, **cacti_kwargs):
    """Scratchpad area in mm2 at process_node.

    Nodes CACTI models natively (22nm and up) use CACTI's area directly. Smaller nodes
    reuse CACTI's 22nm area efficiency with the target bitcell area, which defaults to
    the published bitcell in cost_model.py and can be overridden with bitcell_area_mm2.
    """
    if size_bytes <= 0:
        return 0

    node_nm = _node_nm(process_node)
    if node_nm in CACTI_NATIVE_NODES_NM and bitcell_area_mm2 is None:
        return run_cacti(size_bytes, tech_nm=node_nm, **cacti_kwargs)["area_mm2"]

    if bitcell_area_mm2 is None:
        bitcell_area_mm2 = find_published_sram_bitcell_area(process_node)
    tech_nm = node_nm if node_nm in CACTI_NATIVE_NODES_NM else CACTI_MIN_NODE_NM
    cacti = run_cacti(size_bytes, tech_nm=tech_nm, **cacti_kwargs)
    bits = max(int(size_bytes), CACTI_MIN_SIZE_BYTES) * 8
    return bits * bitcell_area_mm2 / cacti["area_efficiency"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scratchpad area from CACTI")
    parser.add_argument("--kb", type=float, required=True, help="scratchpad capacity in KB")
    parser.add_argument("--node", default="7nm", help="process node, e.g. 3nm, 7nm, 22nm")
    parser.add_argument("--block", type=int, default=64, help="access width in bytes")
    parser.add_argument("--banks", type=int, default=1)
    parser.add_argument("--bitcell-um2", type=float, default=None, help="override target bitcell area")
    args = parser.parse_args()

    size_bytes = int(args.kb * 1024)
    bitcell_mm2 = args.bitcell_um2 / 1e6 if args.bitcell_um2 is not None else None
    node_nm = _node_nm(args.node)
    tech_nm = node_nm if node_nm in CACTI_NATIVE_NODES_NM else CACTI_MIN_NODE_NM
    cacti = run_cacti(size_bytes, block_bytes=args.block, banks=args.banks, tech_nm=tech_nm)
    area = scratchpad_area_mm2(size_bytes, args.node, bitcell_area_mm2=bitcell_mm2,
                               block_bytes=args.block, banks=args.banks)

    print(f"CACTI @ {tech_nm}nm: area {cacti['area_mm2']:.4f} mm2, "
          f"efficiency {cacti['area_efficiency'] * 100:.1f}%, "
          f"bitcell {cacti['bitcell_area_mm2'] * 1e6:.4f} um2, "
          f"access {cacti['access_time_ns']:.3f} ns")
    print(f"Scratchpad {args.kb:g} KB @ {args.node}: {area:.4f} mm2")
