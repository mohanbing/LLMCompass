# Author: August Ning aning@princeton.edu
# Date started: 12 October 2023
# This file is the cost model for Naivesim

import argparse
import json
import math

# lots of parameters required for calculating silicon die area cost

# these are in terms of million transistors per mm2
transistor_density_7nm = 96.3
transistor_density_6nm = 114.2
transistor_density_5nm = 138.2
transistor_density_3nm = 224.0
transistor_density_90nm = 1.6

sram_bit_cell_density_7nm = 1.70e-07
sram_bit_cell_density_6nm = 1.40e-07
sram_bit_cell_density_5nm = 1.25e-07
sram_bit_cell_density_3nm = 1.25e-07

# cache size overheads derived from cacti for cache sizes
# 4096, 8192, 16384, ..., 1 MB
cache_area_efficiency_arr = [0.076, 0.142, 0.247, 0.393, 0.559, \
                             0.704, 0.526, 0.602, 0.561]

# fpu transistor counts are for 64 bit FPU, based off Ariane and OpenPiton's SPARC T1
# assume that fp32 are half the transistors
# int32 transistor count is based off of Ariane's Mult and OpenPiton's SPARC T1
# systolic array is for 1x1 area

# scale FPU area by mantissa bits quadratically
fpu64_transistor_count = 685300
fpu32_transistor_count = fpu64_transistor_count * ((23 / 52) ** 2)
fpu16_transistor_count = fpu64_transistor_count * ((10 / 52) ** 2)

int32_transistor_count = 177690

# based off of A100 SM and MI 210 CU
# these overheads are per sublane, per vector width
# (ex 32 for A100, 16 for MI 210)
per_sublane_control_transistor_count = 996200
nvidia_per_sublane_control_transistor_count = 725650
amd_per_sublane_control_transistor_count = 1534500

per_sublane_control_dict = {'nvidia':per_sublane_control_transistor_count, \
                            'amd':per_sublane_control_transistor_count}

per_core_comm_transistor_count = 44300000
nvidia_per_core_comm_transistor_count = 55000000
amd_per_core_comm_transistor_count = 33600000

per_core_comm_dict = {'nvidia':per_core_comm_transistor_count, \
                      'amd':per_core_comm_transistor_count}


# memory controllers scale with process node, but PHYs do not
# pcie, ddr, hbm
# note: DDR link unit is 32 bits
pcie5_phy_mm2_per_lane = 0.64
pcie4_phy_mm2_per_lane = 0.48
ddr5_phy_mm2_per_link_unit = 1.45
hbm2e_phy_mm2_per_link_unit = 10.45
nvlink3_phy_mm2_per_link_unit = 1.888
nvlink4_phy_mm2_per_link_unit = 0.965
infinity_fabric_phy_mm2_per_link_unit = 5.69

pcie5_ctrl_transistors_per_lane = 5372100
pcie4_ctrl_transistors_per_lane = 3962500
ddr5_ctrl_transistors_per_link_unit = 90446400
hbm2e_ctrl_transistors_per_link_unit = 552743000
nvlink3_ctrl_transistors_per_link_unit = 74632000
nvlink4_ctrl_transistors_per_link_unit = 86628000
infinity_fabric_ctrl_transistors_per_link_unit = 348148000

# mem tech keywords
PCIE5 = 'PCIe5'
PCIE4 = 'PCIe4'
DDR5 = 'DDR5'
HBM = 'HBM2e'
NVLINK3 = 'NVLink3'
NVLINK4 = 'NVLink4'
INFINITYFABRIC = 'InfinityFabric'

# die-to-die (D2D) PHYs between chiplets in a package
# UCIe values are the UCIe 1.0/2.0 key metrics at 32 GT/s (Das Sharma, HOTI 2025):
#   UCIe-S: x32 module @ 110um bump pitch is 1.143 mm (shoreline) x 1.54 mm, 224 GB/s/mm
#   UCIe-A: x64 module @ 55um bump pitch is 0.389 mm (shoreline) x 1.585 mm, 1317 GB/s/mm
# GRS is NVIDIA's 16nm link used in Simba: 81K um2 per 25 Gb/s lane
# (Poulton et al. JSSC 2019, as used by ThermoDSE), shoreline not reported
# bandwidths count both directions. D2D PHYs are bump limited, so like the
# other PHYs their area does not scale with process node
UCIE_S = 'UCIe-S'
UCIE_A = 'UCIe-A'
GRS = 'GRS'

# (module area mm2, module shoreline mm, module bandwidth GB/s, reference data rate GT/s)
d2d_phy_module_dict = {UCIE_S:(1.143 * 1.54, 1.143, 256, 32), \
                       UCIE_A:(0.389 * 1.585, 0.389, 512, 32), \
                       GRS:(0.081, None, 3.125, 25)}

# average via dramexchange spot price, Oct 2023
ddr5_cost_per_gb = 0.30
hbm_cost_per_gb = 7

# return die area for a dimension x dimension SA with a
# give bitwidth FPU at a given process node
# right now, we model each PE's MAC as a FPU
def calc_systolic_array_area_mm2(dimension_x, dimension_y, bitwidth, transistor_density_mil_mm2):
    if bitwidth == 'fp64':
        total_transistor_count = fpu64_transistor_count * dimension_x * dimension_y
    elif bitwidth == 'fp32':
        total_transistor_count = fpu32_transistor_count * dimension_x * dimension_y
    elif bitwidth == 'fp16':
        total_transistor_count = fpu16_transistor_count * dimension_x * dimension_y
    else:
        raise ValueError(f"Unsupported systolic-array data type: {bitwidth}")

    return total_transistor_count / 1e6 / transistor_density_mil_mm2

# vector width corresponds to number of FPUs you have
def calc_vector_area_mm2(int32_count, fp16_count, fp32_count, fp64_count, transistor_density_mil_mm2):
    total_transistor_count = 0
    total_transistor_count += int32_count * int32_transistor_count
    total_transistor_count += fp16_count * fpu16_transistor_count
    total_transistor_count += fp32_count * fpu32_transistor_count
    total_transistor_count += fp64_count * fpu64_transistor_count
    
    return total_transistor_count / 1e6 / transistor_density_mil_mm2

# for cache designs, if the desired capacity is larger than the max cache unit
# split them up into multiple units of the max capacity
# min cache size is 4096 bytes
def calc_cache_sram_area_mm2(capacity_bytes, sram_bitcell_area_mm2, max_cache_unit_bytes=(2**19)):
    if capacity_bytes <= 0:
        return 0

    if capacity_bytes > max_cache_unit_bytes:
        num_cache_units = math.ceil(capacity_bytes / max_cache_unit_bytes)
        unit_size_bytes = max_cache_unit_bytes
    else:
        num_cache_units = 1
        unit_size_bytes = capacity_bytes

    # cache size model is for capacity of 4096 bytes to 1 MB
    if unit_size_bytes < 2 ** 12:
        unit_size_bytes = 2 ** 12

    area_efficiency_index = math.ceil(math.log(unit_size_bytes, 2)) - 12
    area_efficiency_factor = cache_area_efficiency_arr[area_efficiency_index]
    unit_cache_area = unit_size_bytes * 8 * sram_bitcell_area_mm2 / area_efficiency_factor
    cache_area = num_cache_units * unit_cache_area
    return cache_area

# area model comes from EMPIRE
# num_reg_files: how many distinct register files each sublanes has
# D: how many registers there are in each RF
# W: bits per register
# P: number of read/write ports
def calc_reg_file_area(num_reg_files, D, W, P, transistor_density_mil_mm2):
    area_90nm_um2 = (3.29 * 10**4) - (1.09 * 10**3 * D) - (8.83 * 10**2 * W) - (5.55 * 10**3 * P) \
           + (5.35 * 10**1 * D * W) + (1.50 * 10**-2 * D**2) + (1.08 * 10**-2 * W**2) \
           + (5.86 * 10**-1 * P**2) + (1.42 * 10**2 * D * P) + (3.68 * 10**2 * W * P)
    
    # need to convert um2 to mm2, convert to 7nm
    area_90nm_mm2 = area_90nm_um2 / 1e6
    area_mm2 = area_90nm_mm2 * (transistor_density_90nm / transistor_density_mil_mm2)
    total_reg_file_area = num_reg_files * area_mm2
    return total_reg_file_area

# for width, for PCIe and NVLink, it is the whole lane
# for DDR and HBM, it's 128 bits and 1024 bits respectively
def calc_mem_controller_area_mm2(mem_tech, width, transistor_density_mil_mm2):
    controller_transistor_count = -1

    if mem_tech == PCIE5:
        controller_transistor_count = pcie5_ctrl_transistors_per_lane * width
    elif mem_tech == PCIE4:
        controller_transistor_count = pcie4_ctrl_transistors_per_lane * width
    elif mem_tech == DDR5:
        controller_transistor_count = ddr5_ctrl_transistors_per_link_unit * width
    elif mem_tech == HBM:
        controller_transistor_count = hbm2e_ctrl_transistors_per_link_unit * width
    elif mem_tech == NVLINK3:
        controller_transistor_count = nvlink3_ctrl_transistors_per_link_unit * width
    elif mem_tech == NVLINK4:
        controller_transistor_count = nvlink4_ctrl_transistors_per_link_unit * width
    elif mem_tech == INFINITYFABRIC:
        controller_transistor_count = infinity_fabric_ctrl_transistors_per_link_unit * width
    else:
        raise ValueError(f"Unsupported memory controller technology: {mem_tech}")

    return (controller_transistor_count / 1e6) / transistor_density_mil_mm2

def calc_mem_phy_area_mm2(mem_tech, width):
    if mem_tech == PCIE5:
        return pcie5_phy_mm2_per_lane * width
    elif mem_tech == PCIE4:
        return pcie4_phy_mm2_per_lane * width
    elif mem_tech == DDR5:
        return ddr5_phy_mm2_per_link_unit * width
    elif mem_tech == HBM:
        return hbm2e_phy_mm2_per_link_unit * width
    elif mem_tech == NVLINK3:
        return nvlink3_phy_mm2_per_link_unit * width
    elif mem_tech == NVLINK4:
        return nvlink4_phy_mm2_per_link_unit * width
    elif mem_tech == INFINITYFABRIC:
        return infinity_fabric_phy_mm2_per_link_unit * width
    else:
        raise ValueError(f"Unsupported PHY technology: {mem_tech}")

# D2D PHYs come in fixed size modules, so round up to whole modules
# bandwidth_GBps is the total (both directions) D2D bandwidth this die needs
# data rate scales module bandwidth at fixed area, valid up to the spec max (32 GT/s for UCIe 2.0)
def calc_d2d_phy_area_mm2(phy_type, bandwidth_GBps, data_rate_GTs=None):
    if phy_type not in d2d_phy_module_dict:
        raise ValueError(f"Unsupported D2D PHY: {phy_type}")

    module_area, module_shoreline, module_bandwidth, ref_data_rate = d2d_phy_module_dict[phy_type]
    if data_rate_GTs is not None:
        module_bandwidth = module_bandwidth * data_rate_GTs / ref_data_rate

    num_modules = math.ceil(bandwidth_GBps / module_bandwidth)
    shoreline_mm = num_modules * module_shoreline if module_shoreline is not None else None
    return num_modules * module_area, shoreline_mm

def find_logic_sram_transistor_density(process_node):
    if '7' in process_node:
        return transistor_density_7nm, sram_bit_cell_density_7nm
    elif '6' in process_node:
        return transistor_density_6nm, sram_bit_cell_density_6nm
    elif '5' in process_node:
        return transistor_density_5nm, sram_bit_cell_density_5nm
    elif '3' in process_node:
        return transistor_density_3nm, sram_bit_cell_density_3nm

    raise Exception("Invalid Process Node")


# a compute core consists of a fixed control overhead
# a specified width fp32 vector engine
# a specified dimmension fp16 systolic array
# a specified L1 cache
# at a specified process node
# NB: you can fit multiple cores onto a single die for chiplet systems
def calc_compute_chiplet_area_mm2(configs_dict, verbose=False):
    total_die_map = {}
    core_breakdown_map = {}
    device_name = configs_dict['name']
    device_brand = 'nvidia' if 'nvidia' in device_name.lower() else 'amd'
    vector_width = configs_dict['device']['compute_chiplet']['core']['vector_unit']['vector_width']
    vector_int32_count = configs_dict['device']['compute_chiplet']['core']['vector_unit']['int32_count']
    vector_fp16_count = configs_dict['device']['compute_chiplet']['core']['vector_unit']['fp16_count']
    vector_fp32_count = configs_dict['device']['compute_chiplet']['core']['vector_unit']['fp32_count']
    vector_fp64_count = configs_dict['device']['compute_chiplet']['core']['vector_unit']['fp64_count']
    sa_dim_x = configs_dict['device']['compute_chiplet']['core']['systolic_array']['array_width']
    sa_dim_y = configs_dict['device']['compute_chiplet']['core']['systolic_array']['array_height']
    sa_bitwidth = configs_dict['device']['compute_chiplet']['core']['systolic_array']['data_type']
    register_file = configs_dict['device']['compute_chiplet']['core'].get('register_file', {})
    num_reg_files = register_file.get('num_reg_files', 0)
    num_registers = register_file.get('num_registers', 0)
    register_bitwidth = register_file.get('register_bitwidth', 0)
    num_rdwr_ports = register_file.get('num_rdwr_ports', 0)
    sublane_count = configs_dict['device']['compute_chiplet']['core']['sublane_count']
    core_config = configs_dict['device']['compute_chiplet']['core']
    chiplet_config = configs_dict['device']['compute_chiplet']
    if 'L0_SRAM_KB' in core_config:
        l0_cache_size_bytes = core_config['L0_SRAM_KB'] * (2 ** 10)
        l1_cache_size_bytes = chiplet_config.get('L1_SRAM_KB', 0) * (2 ** 10)
        l2_cache_size_bytes = chiplet_config.get('L2_SRAM_KB', 0) * (2 ** 10)
    else:
        l0_cache_size_bytes = 0
        l1_cache_size_bytes = core_config.get('SRAM_KB', 0) * (2 ** 10)
        l2_cache_size_bytes = 0
    process_node = configs_dict['device']['compute_chiplet']['process_node']
    cores_per_chiplet = chiplet_config.get('physical_core_count', chiplet_config.get('core_count'))
    if cores_per_chiplet is None:
        raise KeyError("compute_chiplet requires physical_core_count or core_count")

    # each sublane has a SA and vector unit. a core is made up of sublanes. a chiplet has multiple cores
    transistor_density_mil_mm2, sram_density_bitcell_mm2 = find_logic_sram_transistor_density(process_node)
    per_sublane_area_mm2 = 0
    per_sublane_control_area_mm2 = per_sublane_control_dict[device_brand] / 1e6 / transistor_density_mil_mm2
    per_sublane_area_mm2 += (vector_width * per_sublane_control_area_mm2)
    control_logic_area = per_sublane_area_mm2 * sublane_count

    per_lane_vector_area = calc_vector_area_mm2(vector_int32_count, vector_fp16_count, vector_fp32_count, vector_fp64_count, transistor_density_mil_mm2)
    per_sublane_area_mm2 += per_lane_vector_area

    per_lane_sa_area = calc_systolic_array_area_mm2(sa_dim_x, sa_dim_y, sa_bitwidth, transistor_density_mil_mm2)
    per_sublane_area_mm2 += per_lane_sa_area

    per_lane_regfile_area = calc_reg_file_area(num_reg_files, num_registers, register_bitwidth, num_rdwr_ports, transistor_density_mil_mm2)
    per_sublane_area_mm2 += per_lane_regfile_area

    per_core_compute_area_mm2 = per_sublane_area_mm2 * sublane_count
    l0_cache_area_mm2 = calc_cache_sram_area_mm2(l0_cache_size_bytes, sram_density_bitcell_mm2) * sublane_count
    l1_cache_area_mm2 = calc_cache_sram_area_mm2(l1_cache_size_bytes, sram_density_bitcell_mm2)
    l2_cache_slice_area_mm2 = calc_cache_sram_area_mm2(l2_cache_size_bytes, sram_density_bitcell_mm2)
    per_core_area_mm2 = per_core_compute_area_mm2 + l0_cache_area_mm2 + l1_cache_area_mm2 + l2_cache_slice_area_mm2

    core_breakdown_map['total_core_area'] = per_core_area_mm2
    core_breakdown_map['control_area'] = control_logic_area
    core_breakdown_map['alu_area'] = per_lane_vector_area * sublane_count
    core_breakdown_map['sa_area'] = per_lane_sa_area * sublane_count
    core_breakdown_map['regfile_area'] = per_lane_regfile_area * sublane_count
    core_breakdown_map['l0_buffer_area'] = l0_cache_area_mm2
    core_breakdown_map['l1_buffer_area'] = l1_cache_area_mm2
    core_breakdown_map['l2_buffer_area'] = l2_cache_slice_area_mm2

    total_cores_area = per_core_area_mm2 * cores_per_chiplet
    total_crossbar_area = (per_core_comm_dict[device_brand] / 1e6 / transistor_density_mil_mm2) * cores_per_chiplet
    # each core has an area overhead to connect to the xbar
    compute_chiplet_area_mm2 = total_cores_area + total_crossbar_area

    # optional D2D PHY linking this chiplet to the rest of the package
    d2d_phy_area, d2d_shoreline_mm = 0, 0
    d2d_config = chiplet_config.get('d2d')
    if d2d_config is not None:
        d2d_phy_area, d2d_shoreline_mm = calc_d2d_phy_area_mm2(d2d_config['phy'], \
            d2d_config['bandwidth_byte'] / 1e9, d2d_config.get('data_rate_GTs'))
    compute_chiplet_area_mm2 += d2d_phy_area

    total_die_map['total_area'] = compute_chiplet_area_mm2
    total_die_map['cores_area'] = total_cores_area
    total_die_map['crossbar_area'] = total_crossbar_area
    total_die_map['d2d_phy_area'] = d2d_phy_area
    # compare against the die perimeter (4 * sqrt(area) for a square die); PHYs must sit on the edge
    total_die_map['d2d_shoreline_mm'] = d2d_shoreline_mm
    total_die_map['die_perimeter_mm'] = 4 * math.sqrt(compute_chiplet_area_mm2)
    if verbose:
        return compute_chiplet_area_mm2, core_breakdown_map, total_die_map
    else:
        return compute_chiplet_area_mm2


# NB: for mem_tech, if you are using DDR or HBM, it will be 128 bits and 1024 bits respectively per lane
# for PCIe and NVLink, specify the number of lanes (128bits per lane)
# def calc_io_die_area_mm2(cache_size_bytes, mem_tech, mem_tech_width, num_nvlink_phys, \
#                          transistor_density_mil_mm2, sram_density_bitcell_mm2):
def calc_io_die_area_mm2(config_dict, verbose=False):
    total_die_map = {}

    cache_size_bytes = config_dict['device']['io'].get('physical_global_buffer_MB', 0) * (2 ** 20)
    mem_tech = config_dict['device']['memory_protocol']
    num_mem_tech_units = config_dict['device']['io']['memory_channel_physical_count']
    gpu_gpu_comm_tech = config_dict['interconnect']['link']['name']
    num_gpu_gpu_comm_phy = config_dict['interconnect']['link_count_per_device']
    process_node = config_dict['device']['io']['process_node']

    transistor_density_mil_mm2, sram_density_bitcell_mm2 = find_logic_sram_transistor_density(process_node)
    io_die_area_mm2 = 0
    io_die_area_mm2 += calc_cache_sram_area_mm2(cache_size_bytes, sram_density_bitcell_mm2)
    global_buffer_area = io_die_area_mm2

    # mem tech for communicating to off chip memory
    mem_phy_area = calc_mem_phy_area_mm2(mem_tech, num_mem_tech_units)
    mem_controller_area = calc_mem_controller_area_mm2(mem_tech, num_mem_tech_units, transistor_density_mil_mm2)
    io_die_area_mm2 += mem_phy_area
    io_die_area_mm2 += mem_controller_area

    # every  IO die has a few NV links for chip to chip communication
    device_phy_area = calc_mem_phy_area_mm2(gpu_gpu_comm_tech, num_gpu_gpu_comm_phy)
    device_controller_area = calc_mem_controller_area_mm2(gpu_gpu_comm_tech, num_gpu_gpu_comm_phy, transistor_density_mil_mm2)
    io_die_area_mm2 += device_phy_area
    io_die_area_mm2 += device_controller_area

    # the IO die terminates the D2D link of every compute chiplet
    d2d_phy_area = 0
    d2d_config = config_dict['device']['compute_chiplet'].get('d2d')
    if d2d_config is not None:
        compute_chiplet_count = config_dict['device'].get('compute_chiplet_count', 1)
        d2d_phy_area, _ = calc_d2d_phy_area_mm2(d2d_config['phy'], \
            compute_chiplet_count * d2d_config['bandwidth_byte'] / 1e9, d2d_config.get('data_rate_GTs'))
    io_die_area_mm2 += d2d_phy_area

    total_die_map['total_die_area'] = io_die_area_mm2
    total_die_map['global_buffer_area'] = global_buffer_area
    total_die_map['mem_phy_area'] = mem_phy_area
    total_die_map['mem_controller_area'] = mem_controller_area
    total_die_map['device_phy_area'] = device_phy_area
    total_die_map['device_controller_area'] = device_controller_area
    total_die_map['d2d_phy_area'] = d2d_phy_area

    if verbose:
        return io_die_area_mm2, total_die_map
    else:
        return io_die_area_mm2

def calc_device_area_mm2(config_dict, verbose=False):
    """Estimate compute-chiplet area plus one IO-die area for a device."""
    compute_chiplet_area, core_breakdown, compute_breakdown = calc_compute_chiplet_area_mm2(
        config_dict, verbose=True
    )
    io_die_area, io_breakdown = calc_io_die_area_mm2(config_dict, verbose=True)
    compute_chiplet_count = config_dict['device'].get('compute_chiplet_count', 1)
    total_compute_area = compute_chiplet_area * compute_chiplet_count
    total_device_area = total_compute_area + io_die_area

    if verbose:
        return {
            'compute_chiplet_area_mm2': compute_chiplet_area,
            'compute_chiplet_count': compute_chiplet_count,
            'total_compute_area_mm2': total_compute_area,
            'io_die_area_mm2': io_die_area,
            'total_device_area_mm2': total_device_area,
            'core_breakdown': core_breakdown,
            'compute_chiplet_breakdown': compute_breakdown,
            'io_die_breakdown': io_breakdown,
        }
    return total_device_area


def main():
    parser = argparse.ArgumentParser(description='Estimate device area from an architecture JSON file.')
    parser.add_argument('config', help='Path to the architecture JSON config')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config_dict = json.load(config_file)

    area = calc_device_area_mm2(config_dict, verbose=True)
    print(f"Compute chiplet area: {area['compute_chiplet_area_mm2']:.3f} mm^2 each")
    print(f"Compute chiplets: {area['compute_chiplet_count']}")
    print(f"Total compute area: {area['total_compute_area_mm2']:.3f} mm^2")
    print(f"IO die area: {area['io_die_area_mm2']:.3f} mm^2 (one IO die assumed)")
    print(f"Total device area: {area['total_device_area_mm2']:.3f} mm^2")


if __name__ == '__main__':
    main()