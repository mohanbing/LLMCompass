from software_model.operators import (
    Operator,
    Reshape
)
from software_model.matmul import Matmul

from software_model.utils import Tensor, DataType, data_type_dict
from software_model.graph import DependencyGraph
from software_model.utils import SymbolTable
from math import floor, prod


class Conv2d(Operator):
    __count = 0
    def __init__(self, in_channels: int, out_channels:int, kernel_size:int, stride:int, bias:bool, padding:int, data_type):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight_bias = Tensor([self.out_channels, self.in_channels, kernel_size, kernel_size], data_type)
        self.data_type = data_type
        self.name = f"{self.__class__.__name__}_{Conv2d.__count}"
        self.desc = "Conv"

        Conv2d.__count += 1
    
    def __call__ (self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        h_out = floor((h + 2*self.padding - self.kernel_size)/ self.stride + 1)
        w_out = floor((w + 2*self.padding - self.kernel_size)/ self.stride + 1)

        output = Tensor([b, self.out_channels, h_out, w_out], self.data_type)
        DependencyGraph.add_node_to_graph(output, [x, self.weight_bias], self.__class__.__name__, self.name, self.desc, stride=self.stride)
        return output
    
class Linear(Operator):
    __count = 0
    def __init__(self, in_feat: int, out_feat:int, bias:bool, data_type):
        self.in_features = in_feat
        self.out_features = out_feat
        if bias:
            self.weight_bias = Tensor([self.in_features+1, self.out_features], data_type)
        else:
            self.weight_bias = Tensor([self.in_features, self.out_features], data_type)
        self.data_type = data_type
        self.name = f"{self.__class__.__name__}_{Linear.__count}"

        Linear.__count += 1
    
    def __call__ (self, x: Tensor) -> Tensor:
        output = Matmul(data_type=self.data_type)(x, self.weight_bias)
        return output
    

class ReLU(Operator):
    __count = 0
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None
        self.name = f"{self.__class__.__name__}_{ReLU.__count}"
        ReLU.__count += 1

    def __call__(self, x: Tensor) -> Tensor:
        assert self.data_type == x.data_type
        self.shape = x.shape
        DependencyGraph.add_node_to_graph(x, [x], self.__class__.__name__, self.name, core=self.core_device)
        return x
    
class MaxPool2d(Operator):
    __count = 0
    def __init__(self, kernel_size:int, stride:int, padding:int, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.name = f"{self.__class__.__name__}_{MaxPool2d.__count}"
        MaxPool2d.__count += 1

    def __call__(self, x: Tensor) -> Tensor:
        assert self.data_type == x.data_type
        b, c, h, w = x.shape

        h_out = floor((h + 2*self.padding - self.kernel_size)/ self.stride + 1)
        w_out = floor((w + 2*self.padding - self.kernel_size)/ self.stride + 1)

        output = Tensor([b, c, h_out, w_out], self.data_type)
        DependencyGraph.add_node_to_graph(output, [x], self.__class__.__name__, self.name, core=self.core_device, stride=self.stride)
        return output
    

class AlexNet(Operator):
    def __init__(self, data_type):
        super().__init__(0, 0, 0, 0, data_type)
        self.data_type = data_type

        self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, bias=False, padding=2, data_type=data_type)
        self.relu1 = ReLU(data_type)
        self.max_pool1 = MaxPool2d(kernel_size=3, stride=2, padding=0, data_type=data_type)

        self.conv2 = Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, bias=False, padding=2, data_type=data_type)
        self.relu2 = ReLU(data_type)
        self.max_pool2 = MaxPool2d(kernel_size=3, stride=2, padding=0, data_type=data_type)

        self.conv3 = Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, bias=False, padding=1, data_type=data_type)
        self.relu3 = ReLU(data_type)

        self.conv4 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, bias=False, padding=1, data_type=data_type)
        self.relu4 = ReLU(data_type)

        self.conv5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=False, padding=1, data_type=data_type)
        self.relu5 = ReLU(data_type)
        self.max_pool3 = MaxPool2d(kernel_size=3, stride=2, padding=0, data_type=data_type)

        self.linear1 = Linear(in_feat=256, out_feat=4096, bias=False, data_type=data_type)
        self.relu6 = ReLU(data_type)

        self.linear2 = Linear(in_feat=4096, out_feat=4096, bias=False, data_type=data_type)
        self.relu7 = ReLU(data_type)

        self.linear3 = Linear(in_feat=4096, out_feat=10, bias=False, data_type=data_type)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.max_pool3(x)

        x = Reshape(self.data_type)(x, [x.shape[0], prod(x.shape[1:])]) # flatten
        x = self.linear1(x)
        x = self.relu6(x)

        x = self.linear2(x)
        x = self.relu7(x)

        x = self.linear3(x)
        return x

if __name__ == "__main__":
    from pathlib import Path

    batch_size = 1
    
    model = AlexNet(
        data_type=data_type_dict["fp16"]
    )
    x = Tensor([batch_size, 3, 28, 28], data_type_dict["fp16"])
    logits = model(x)

    symbol_table_path = Path("symbol_table.json")
    dep_graph_path = Path("dep_graph_alexnet.json")
    
    SymbolTable.dump_symbol_table_to_json(symbol_table_path)
    DependencyGraph.dump_graph_to_json(dep_graph_path)
    total_params = DependencyGraph.get_learnable_parameters()

    print("symbol table dumped to: ", symbol_table_path)
    print("dep graph dumped to: ", dep_graph_path)
    print(f"Learnable Parameter Count: {total_params}")