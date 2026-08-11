from software_model.alexnet import Conv2d
from software_model.utils import SymbolTable
from software_model.graph import DependencyGraph
from software_model.utils import Tensor, data_type_dict

def conv3x3(in_c:int, out_c:int, stride:int=1, data_type=data_type_dict["fp16"]) -> Conv2d:
    return Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                  padding=1, bias=False, data_type=data_type)

if __name__ == "__main__":
    from pathlib import Path

    batch_size = 1
    
    model = conv3x3(3, 32, stride=2)
    x = Tensor([batch_size, 3, 28, 28], data_type_dict["fp16"])
    logits = model(x)

    symbol_table_path = Path("symbol_table.json")
    dep_graph_path = Path("dep_graph_single_conv.json")
    
    SymbolTable.dump_symbol_table_to_json(symbol_table_path)
    DependencyGraph.dump_graph_to_json(dep_graph_path)
    total_params = DependencyGraph.get_learnable_parameters()

    print("symbol table dumped to: ", symbol_table_path)
    print("dep graph dumped to: ", dep_graph_path)
    print(f"Learnable Parameter Count: {total_params}")