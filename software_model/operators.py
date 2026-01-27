from utils import size, closest_factors
from typing import List, Optional
from hardware_model.device import Device
from software_model.utils import Tensor, DataType
from software_model.graph import DependencyGraph


class Operator:
    def __init__(
        self,
        flop_count,
        load_count,
        store_count,
        peak_memory_usage,
        data_type: DataType,
        gpu_device=None,
        verbose=True,
        core_device:int = -1,
    ):
        self.flop_count = flop_count
        self.load_count = load_count
        self.store_count = store_count
        self.io_count = load_count + store_count
        self.peak_memory_usage = peak_memory_usage
        self.data_type = data_type
        self.gpu_device = gpu_device
        self.verbose = verbose
        self.log = ""
        self.comment = ""
        # simulation results
        self.latency = 0
        self.latency_on_gpu = 1
        self.is_io_bound = None
        # run on gpu
        self.iterations = 50
        self.dependencies = []
        self.desc = ""
        self.core_device = core_device

    def set_desc(self, desc:str):
        self.desc = desc

    def set_core_device(self, core_device:int):
        self.core_device = core_device

    class mapping:
        pass


# auxilary functions


class Reshape(Operator):
    __count = 0

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.name = f"{self.__class__.__name__}_{Reshape.__count}"
        self.input_shape = None
        self.output_shape = None
        Reshape.__count += 1

    def __call__(self, input: Tensor, output_shape: List[int]) -> Tensor:
        assert input.size == size(output_shape)
        self.flop_count = 0
        self.load_count = 0
        self.store_count = 0
        self.io_count = 0
        self.peak_memory_usage = 0
        self.input_shape = input.shape
        self.output_shape = output_shape
        output = Tensor(output_shape, self.data_type, input)
        output.set_desc(input.desc)
        DependencyGraph.add_node_to_graph(output, [input], self.__class__.__name__, self.name, core=self.core_device)
        return output


class Concat(Operator):
    __count=0

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.name = f"{self.__class__.__name__}_{Concat.__count}"
        self.input1_shape = None
        self.input2_shape = None
        self.concat_dim = None
        self.output_shape = None
        Concat.__count += 1

    def __call__(self, input1: Tensor, input2: Tensor, concat_dim: int, output:Optional[Tensor] = None) -> Tensor:
        assert len(input1.shape) == len(input2.shape)
        for i in range(len(input1.shape)):
            if i != concat_dim:
                assert input1.shape[i] == input2.shape[i]
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.concat_dim = concat_dim
        self.flop_count = 0
        self.load_count = input1.size + input2.size
        self.store_count = input1.size + input2.size
        self.io_count = self.load_count + self.store_count
        self.peak_memory_usage = (input1.size + input2.size) * 2
        self.output_shape = (
            input1.shape[:concat_dim]
            + [input1.shape[concat_dim] + input2.shape[concat_dim]]
            + input1.shape[concat_dim + 1 :]
        )

        if output:
            assert output.shape == self.output_shape
            output = Tensor(shape=output.shape, data_type=output.data_type, reuse_t=output)
        else:
            output = Tensor(self.output_shape, self.data_type)
        
        output.set_desc(input1.desc + "_" + input2.desc)
        DependencyGraph.add_node_to_graph(output, [input1, input2], self.__class__.__name__, self.name, core=self.core_device)
        return output


class Transpose(Operator):
    __count=0

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.name = f"{self.__class__.__name__}_{Transpose.__count}"
        self.input_shape = None
        self.output_shape = None
        Transpose.__count += 1

    def __call__(self, input: Tensor, permute: List[int]) -> Tensor:
        assert len(input.shape) == len(permute)
        self.input_shape = input.shape
        self.permute = permute

        self.flop_count = 0
        self.load_count = size(input.shape)
        self.store_count = self.load_count
        self.io_count = self.load_count + self.store_count
        self.peak_memory_usage = input.size * 2

        self.output_shape = [self.input_shape[i] for i in permute]
        output = Tensor(self.output_shape, self.data_type, input)
        output.set_desc(input.desc)
        DependencyGraph.add_node_to_graph(output, [input], self.__class__.__name__, self.name, core=self.core_device)
        return output
    

class ElementWiseAddition(Operator):
    __count=0

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.name = f"{self.__class__.__name__}_{ElementWiseAddition.__count}"
        self.input_shape = None
        self.output_shape = None
        ElementWiseAddition.__count += 1

    def __call__(self, input1: Tensor, input2: Tensor, output: Optional[Tensor] = None) -> Tensor:
        assert input1.shape == input2.shape
        self.input_shape = input1.shape
        self.output_shape = self.input_shape

        if output:
            assert output.shape == self.output_shape
            new_output = Tensor(shape=output.shape, data_type=output.data_type, reuse_t=output)
            DependencyGraph.add_node_to_graph(new_output, [input1, input2], self.__class__.__name__, self.name, core=self.core_device)
            return new_output
        else:
            output = Tensor(self.output_shape, self.data_type)

        DependencyGraph.add_node_to_graph(output, [input1, input2], self.__class__.__name__, self.name, core=self.core_device)
        return output

class BarrierSync(Operator):
    __count = 0

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.name = f"{self.__class__.__name__}_{BarrierSync.__count}"
        self.input_shape = None
        self.output_shape = None
        BarrierSync.__count += 1
    
    def __call__(self):
        DependencyGraph.add_node_to_graph(None, None, self.__class__.__name__, self.name)
        return None