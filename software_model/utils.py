import json
from pathlib import Path
from typing import List, NamedTuple
from utils import size

class SymbolTable:
    addr = -1
    table = {}
    keys = ["variable_name", "start_addr", "data_type", "size", "shape"]

    @classmethod
    def create_entry(cls, t, reuse_t=None):
        entry = {}
        entry["data_type"] = t.data_type
        entry["size"] = t.size * t.data_type.word_size
        entry["shape"] = t.shape

        if reuse_t:
            entry["start_addr"] = cls.table[reuse_t.name]["start_addr"]

        else:
            entry["start_addr"] = SymbolTable.addr + 1
            SymbolTable.addr += 1 + entry["size"]

        entry["variable_name"] = t.name
        cls.table[t.name] = entry
    
    @classmethod
    def lookup(cls, name:str) -> dict:
        return cls.table.get(name)
    
    @classmethod
    def dump_symbol_table_to_json(cls, path:Path):
        with open(path, 'w') as fp:
            json.dump(cls.table, fp)


class DataType(NamedTuple):
    name:str
    word_size:int

data_type_dict = {"int8": DataType("int8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4)}

class Tensor:
    __count = 0

    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"]
    ) -> None:
        self.name = f"{self.__class__.__name__}_{Tensor.__count}"
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type

        SymbolTable.create_entry(self)
        Tensor.__count += 1