import json
import pandas as pd
from typing import List, Optional
from pathlib import Path

from software_model.utils import Tensor
from software_model.utils import SymbolTable

class DependencyGraph:
    graph = {}

    @classmethod
    def add_node_to_graph(cls, target: Optional[Tensor], dep_list:Optional[List[Tensor]], op:str, 
                          op_name: str, op_desc: str = None, 
                          core:int = -1, 
                          batched_matmul_details: dict = None,
                          sharded_matmul_details: dict = None,
                          stride: int = 1):
        
        if batched_matmul_details:
            cls.graph[op_name] = dict(
                dep = {f"op_{op_c}":SymbolTable.table[dep.name] for op_c, dep in enumerate(dep_list)},
                op = op,
                out = SymbolTable.table[target.name],
                op_desc = op_desc,
                core = str(core),
                chiplet = str(0),
                batched_matmul_offsets = batched_matmul_details,
                stride=stride
            )
        else:
            if sharded_matmul_details:
                cls.graph[op_name] = dict(
                    dep = {f"op_{op_c}":SymbolTable.table[dep.name] for op_c, dep in enumerate(dep_list)},
                    op = op,
                    out = SymbolTable.table[target.name],
                    op_desc = op_desc,
                    core = str(core),
                    chiplet = str(0),
                    sharded_matmul_offsets = sharded_matmul_details,
                    stride=stride
                )
            else:
                if (target is None) and (dep_list is None):
                    cls.graph[op_name] = dict(
                    dep = None,
                    op = op,
                    out = None,
                    op_desc = op_desc,
                    core = str(core),
                    chiplet = str(0),
                    stride=stride
                )
                else:
                    cls.graph[op_name] = dict(
                        dep = {f"op_{op_c}":SymbolTable.table[dep.name] for op_c, dep in enumerate(dep_list)},
                        op = op,
                        out = SymbolTable.table[target.name],
                        op_desc = op_desc,
                        core = str(core),
                        chiplet = str(0),
                        stride=stride
                    )

    @classmethod
    def dump_graph_to_json(cls, path:Path):
        with open(path, 'w') as fp:
            json.dump(cls.graph, fp)

    @classmethod
    def get_learnable_parameters(cls):
        total_params = 0
        for _, tensor_dict in cls.graph.items():
            dep = tensor_dict["dep"]
            op = tensor_dict["op"]
            if op == "Matmul":
                op1 = dep["op_1"]
                op1_shape = op1["shape"]
                prod = 1
                for dim in op1_shape:
                    prod *= dim
                total_params += prod
        
        return total_params
    
    @classmethod
    def reset_and_dump_graph(cls, path:Path):
        DependencyGraph.dump_graph_to_json(path=path)
        DependencyGraph.graph.clear()
    
    # @classmethod
    # def construct_wyvern_topology(cls, path:Path):
    #     topology = []
    #     node_count = 0
    #     for key in cls.graph:
    #         row_dict = {}
    #         row_dict["NODE"] = f"node_{node_count}"
    #         dep = cls.graph[key]["dep"]
    #         for i, tensor_key in enumerate(dep):
    #             if i==0 and len(dep) > 1:
    #                 row_dict["M"] = dep[tensor_key]["shape"][1]
    #                 row_dict["K"] = dep[tensor_key]["shape"][2]

    #         node_count += 1
