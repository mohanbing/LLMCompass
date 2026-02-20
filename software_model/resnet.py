from software_model.alexnet import *
from software_model.operators import ElementWiseAddition, Sequential

from typing import List, Callable, Optional

def conv3x3(in_c:int, out_c:int, stride:int=1, data_type=data_type_dict["fp16"]) -> Conv2d:
    return Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                  padding=1, bias=False, data_type=data_type)

def conv1x1(in_c:int, out_c:int, stride:int=1, data_type=data_type_dict["fp16"]) -> Conv2d:
    return Conv2d(in_c, out_c, kernel_size=1, stride=stride, 
                  padding=0, bias=False, data_type=data_type)

class AdaptiveAvgPool2d(Operator):
    __count = 0
    def __init__(self, output_shape: tuple, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.name = f"{self.__class__.__name__}_{AdaptiveAvgPool2d.__count}"
        self.data_type == data_type
        self.output_shape = output_shape
        AdaptiveAvgPool2d.__count += 1

    def __call__(self, x: Tensor) -> Tensor:
        output = Tensor([x.shape[0], x.shape[1], self.output_shape[0], self.output_shape[1]], self.data_type)
        DependencyGraph.add_node_to_graph(output, [x], self.__class__.__name__, self.name, core=self.core_device)
        return output

class BatchNorm(Operator):
    __count = 0
    def __init__(self, num_features:int, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.name = f"{self.__class__.__name__}_{BatchNorm.__count}"
        self.shape = None
        self.data_type = data_type
        BatchNorm.__count += 1

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        
        DependencyGraph.add_node_to_graph(input, [input], self.__class__.__name__, self.name, core=self.core_device)
        return input

class BasicBlock(Operator):
    expansion = 1

    def __init__(self, in_channels:int, out_channels:int, stride:int=1, 
                 downsample:Optional[Callable] = None, data_type = data_type_dict["fp16"]):
        
        super().__init__(0, 0, 0, 0, data_type)
        self.conv1 = conv3x3(in_channels, out_channels, stride, data_type=data_type)
        self.bn1 = BatchNorm(out_channels, data_type)
        self.relu = ReLU()

        self.conv2 = conv3x3(out_channels, out_channels, data_type=data_type)
        self.bn2 = BatchNorm(out_channels, data_type)

        self.downsample = downsample
        self.data_type = data_type

    def __call__(self, x:Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = ElementWiseAddition(self.data_type)(identity, out, out)
        out = self.relu(out)
        return out
    
class Bottleneck(Operator):
    expansion = 4

    def __init__(self, in_channels:int, out_channels:int, stride:int = 1, 
                 downsample: Optional[Callable] = None, data_type = data_type_dict["fp16"]):
        
        super().__init__(0, 0, 0, 0, data_type)
        self.conv1 = conv1x1(in_channels, out_channels, data_type=data_type)
        self.bn1 = BatchNorm(out_channels, data_type=data_type)

        self.conv2 = conv3x3(out_channels, out_channels, stride, data_type=data_type)
        self.bn2 = BatchNorm(out_channels, data_type=data_type)

        self.conv3 = conv1x1(out_channels, out_channels * self.expansion, data_type=data_type)
        self.bn3 = BatchNorm(out_channels * self.expansion, data_type=data_type)

        self.relu = ReLU(data_type=data_type)
        self.downsample = downsample

    def __call__(self, x:Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = ElementWiseAddition(self.data_type)(identity, out, out)
        out = self.relu(out)
        return out

class ResNet(Operator):
    def __init__(self, block:Bottleneck, layer_list: List[int], num_classes:int, num_channels:int, data_type = data_type_dict["fp16"]):
        self.in_channels = 64
        self.data_type = data_type

        self.conv1 = Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, data_type=data_type)
        self.bn1 = BatchNorm(64, data_type)
        self.relu = ReLU(data_type)
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1, data_type=self.data_type)

        self.layer1 = self._make_layer(block, layer_list[0], planes=64)
        self.layer2 = self._make_layer(block, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(block, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(block, layer_list[3], planes=512, stride=2)

        self.avgpool = AdaptiveAvgPool2d((1,1), data_type=data_type)
        self.fc = Linear(512 * block.expansion, num_classes, bias=False, data_type=data_type)
        
        self.data_type = data_type
    
    def _make_layer(self, block:Bottleneck, blocks:int, planes:int, stride:int = 1):
        ii_downsample = None
        layers: List[Callable] = []
        
        if stride != 1 or self.in_channels != planes * block.expansion:
            ii_downsample = Sequential([Conv2d(self.in_channels, planes*block.expansion, kernel_size=1, stride=stride, padding=0, bias=False, data_type=self.data_type),
                                        BatchNorm(planes*block.expansion, self.data_type)], 
                                        self.data_type)
            
        layers.append(block(self.in_channels, planes, downsample=ii_downsample, stride=stride))
        self.in_channels = planes * block.expansion
        
        for _ in range(blocks-1):
            layers.append(block(self.in_channels, planes))
            
        return Sequential(layers, self.data_type)


    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = Reshape(self.data_type)(x, [x.shape[0], prod(x.shape[1:])]) # flatten
        x = self.fc(x)
        
        return x

def ResNet18(num_classes:int, channels:int = 3, data_type = data_type_dict["fp16"]):
    return ResNet(Bottleneck, [2,2,2,2], num_classes, channels, data_type=data_type)

def ResNet34(num_classes:int, channels:int = 3, data_type = data_type_dict["fp16"]):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels, data_type=data_type)


if __name__ == "__main__":
    from pathlib import Path

    batch_size = 1
    
    model = ResNet18(10, 3)
    x = Tensor([batch_size, 3, 224, 224], data_type_dict["fp16"])
    logits = model(x)

    symbol_table_path = Path("symbol_table.json")
    dep_graph_path = Path("dep_graph_resnet_18.json")
    
    SymbolTable.dump_symbol_table_to_json(symbol_table_path)
    DependencyGraph.dump_graph_to_json(dep_graph_path)
    total_params = DependencyGraph.get_learnable_parameters()

    print("symbol table dumped to: ", symbol_table_path)
    print("dep graph dumped to: ", dep_graph_path)
    print(f"Learnable Parameter Count: {total_params}")