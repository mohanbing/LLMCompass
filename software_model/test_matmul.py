from software_model.alexnet import Matmul
from software_model.utils import SymbolTable
from software_model.operators import Prefetch, AllChipletBarrierSync
from software_model.graph import DependencyGraph
from software_model.utils import Tensor, data_type_dict

def workload(x: Tensor, data_type=data_type_dict["fp16"]) -> Tensor:
    W1 = Tensor([512, 512], data_type=data_type)
    W2 = Tensor([512, 512], data_type=data_type)
    W3 = Tensor([512, 512], data_type=data_type)
    W4 = Tensor([512, 512], data_type=data_type)

    prefetch_obj = Prefetch(data_type=data_type)
    prefetch_obj.set_desc("prefetch_obj")
    prefetch_obj.set_core_device(0)

    # _ = prefetch_obj(x)
    # _ = AllChipletBarrierSync(data_type=data_type)()
    
    matmul1 = Matmul(data_type=data_type)
    matmul1.set_core_device(0)

    matmul2 = Matmul(data_type=data_type)
    matmul2.set_core_device(1)

    matmul3 = Matmul(data_type=data_type)
    matmul3.set_core_device(2)

    matmul4 = Matmul(data_type=data_type)
    matmul4.set_core_device(3)

    y = matmul1(x, W1)
    y = matmul2(x, W2)
    y = matmul3(x, W3)
    y = matmul4(x, W4)

    return y

if __name__ == "__main__":
    from pathlib import Path

    batch_size = 1
    
    x = Tensor([512, 512], data_type_dict["fp16"])
    logits = workload(x, data_type=data_type_dict["fp16"])

    symbol_table_path = Path("symbol_table.json")
    dep_graph_path = Path("4_matmul_no_prefetch.json")
    
    SymbolTable.dump_symbol_table_to_json(symbol_table_path)
    DependencyGraph.dump_graph_to_json(dep_graph_path)
    total_params = DependencyGraph.get_learnable_parameters()

    print("symbol table dumped to: ", symbol_table_path)
    print("dep graph dumped to: ", dep_graph_path)
    print(f"Learnable Parameter Count: {total_params}")