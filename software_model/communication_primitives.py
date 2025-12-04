from hardware_model.device import Device
from hardware_model.interconnect import (
    LinkModule,
    InterConnectModule,
    TopologyType,
    interconnect_module_dict,
)
from software_model.utils import Tensor, DataType
from typing import Any, List
from utils import size
from math import ceil
from software_model.graph import DependencyGraph


class CommunicationPrimitive:
    def __init__(self, data_type: DataType) -> None:
        self.data_type = data_type
        # simulation results
        self.latency = None
        self.core_device = -1
    
    def set_core_device(self, core_device:int):
        self.core_device = core_device


class AllReduceMultiPCB(CommunicationPrimitive):
    __count = 0
    def __init__(self, data_type: DataType) -> None:
        super().__init__(data_type)
        self.name = f"{self.__class__.__name__}_{AllReduceMultiPCB.__count}"
        AllReduceMultiPCB.__count += 1

    def __call__(self, tensors: List[Tensor]) -> Any:
        assert tensors[0].data_type == self.data_type
        b, _, m, n = tensors[0].shape
        target_tensor_shape = [b, len(tensors), m, n]
        target_tensor = Tensor(target_tensor_shape, tensors[0].data_type, reuse_t=tensors[0])
        DependencyGraph.add_node_to_graph(target_tensor, tensors, self.__class__.__name__, self.name, core=self.core_device)
        return target_tensor

    def simulate(self, interconnect_module: InterConnectModule) -> None:
        device_count = interconnect_module.device_count
        link_bandwidth_per_direction = (
            interconnect_module.link_module.bandwidth_per_direction
        )
        link_bandwidth_both_direction = (
            interconnect_module.link_module.bandwidth_both_direction
        )
        link_latency = interconnect_module.link_module.latency
        flit_size = interconnect_module.link_module.flit_size
        header_size = interconnect_module.link_module.header_size
        max_payload_size = interconnect_module.link_module.max_payload_size
        link_count_per_device = interconnect_module.link_count_per_device
        data_size = size(self.input_shape) * self.data_type.word_size
        if interconnect_module.topology == TopologyType.FC:
            edge_bandwidth_per_direction = (
                link_bandwidth_per_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_bandwidth_both_direction = (
                link_bandwidth_both_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_latency = link_latency
            data_size_per_device = data_size / device_count
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            # stage 1: ring reduce
            latency = (
                edge_latency
                + effective_data_size_per_device / edge_bandwidth_both_direction
            ) * (device_count - 1)
            # stage 2: broadcast
            latency += effective_data_size_per_device / edge_bandwidth_per_direction
            latency += (
                data_size / interconnect_module.internal_link_bandwidth_per_direction
            )
            self.latency = latency
            return latency
        elif interconnect_module.topology == TopologyType.RING:
            edge_bandwidth = link_bandwidth_per_direction * link_count_per_device
            edge_latency = link_latency
            data_size_per_device = data_size / device_count
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            per_transmission_latency = effective_data_size_per_device / edge_bandwidth
            latency = (edge_latency + per_transmission_latency) * (
                (device_count - 1) * 2
            )
            latency += (
                data_size / interconnect_module.internal_link_bandwidth_per_direction
            )
            self.latency = latency
        else:
            raise NotImplementedError
        return self.latency
    
class AllGather(CommunicationPrimitive):
    __count = 0
    def __init__(self, data_type: DataType) -> None:
        super().__init__(data_type)
        self.name = f"{self.__class__.__name__}_{AllGather.__count}"
        AllGather.__count += 1

    def __call__(self, tensors: List[Tensor]) -> Any:
        assert tensors[0].data_type == self.data_type
        b, _, m, n = tensors[0].shape
        target_tensor_shape = [b, len(tensors), m, n]
        target_tensor = Tensor(target_tensor_shape, tensors[0].data_type, reuse_t=tensors[0])
        DependencyGraph.add_node_to_graph(target_tensor, tensors, self.__class__.__name__, self.name, core=self.core_device)
        return target_tensor


# class P2P:
#     def __init__(self):
#         self.src = None
#         self.dst = None
#         self.tensor = None

#     def __call__(self, src: int, dst: int, tensor: Tensor):
#         self.src = src
#         self.dst = dst
#         self.tensor = tensor

#     def __simulate__(self, src: ChipletModule, dst: ChipletModule, link: LinkModule):
#         pass


class Broadcast:
    def __init__(self):
        self.src = None
        self.tensor = None

    def __call__(self, src: int, tensor: Tensor):
        self.src = src
        self.tensor = tensor

