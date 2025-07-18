import json
import pandas as pd
from typing import List
from pathlib import Path

from software_model.utils import Tensor
from software_model.utils import SymbolTable

class DependencyGraph:
    graph = {}

    @classmethod
    def add_node_to_graph(cls, target: Tensor, dep_list:List[Tensor], op:str, op_name: str):
        cls.graph[op_name] = dict(
            dep = {f"op_{op_c}":SymbolTable.table[dep.name] for op_c, dep in enumerate(dep_list)},
            op = op,
            out = SymbolTable.table[target.name]
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
