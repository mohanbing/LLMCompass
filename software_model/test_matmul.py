from software_model.alexnet import Matmul
from software_model.utils import SymbolTable
from software_model.graph import DependencyGraph
from software_model.utils import Tensor, data_type_dict

def workload(x: Tensor, data_type=data_type_dict["fp16"]) -> Tensor:
    W1 = Tensor([128, 128], data_type=data_type)
    W2 = Tensor([128, 128], data_type=data_type)
    W3 = Tensor([128, 128], data_type=data_type)

    matmul1 = Matmul(data_type=data_type)
    matmul1.set_core_device(0)

    matmul2 = Matmul(data_type=data_type)
    matmul2.set_core_device(0)

    # matmul3 = Matmul(data_type=data_type)
    # matmul3.set_core_device(0)

    x = matmul1(x, W1)
    x = matmul2(x, W2)
    # x = matmul3(x, W3)

    return x

if __name__ == "__main__":
    from pathlib import Path

    batch_size = 1
    
    x = Tensor([128, 128], data_type_dict["fp16"])
    logits = workload(x, data_type=data_type_dict["fp16"])

    symbol_table_path = Path("symbol_table.json")
    dep_graph_path = Path("dep_graph_single_matmul.json")
    
    SymbolTable.dump_symbol_table_to_json(symbol_table_path)
    DependencyGraph.dump_graph_to_json(dep_graph_path)
    total_params = DependencyGraph.get_learnable_parameters()

    print("symbol table dumped to: ", symbol_table_path)
    print("dep graph dumped to: ", dep_graph_path)
    print(f"Learnable Parameter Count: {total_params}")