import json
from pathlib import Path
from typing import List, NamedTuple
from utils import size

class SymbolTable:
    addr = -1
    table = {}
    keys = ["variable_name", "start_addr", "data_type", "size", "shape"]

    @classmethod
    def create_entry(cls, t, reuse_t=None, offset=0):
        entry = {}
        entry["data_type"] = t.data_type
        entry["size"] = (t.size * t.data_type.word_size)//4
        entry["shape"] = t.shape
        entry["tensor_desc"] = t.desc

        if reuse_t:
            entry["start_addr"] = cls.table[reuse_t.name]["start_addr"] + offset

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

    @classmethod
    def update_tensor_desc(cls, name:str, desc:str):
        cls.table[name]["tensor_desc"] = desc


class DataType(NamedTuple):
    name:str
    word_size:int

data_type_dict = {"int8": DataType("int8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4)}

class Tensor:
    __count = 0

    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"], reuse_t = None
    ) -> None:
        self.name = f"{self.__class__.__name__}_{Tensor.__count}"
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type
        self.desc = ""

        SymbolTable.create_entry(self, reuse_t=reuse_t)
        Tensor.__count += 1
    
    def set_desc(self, desc:str):
        self.desc = desc
        SymbolTable.update_tensor_desc(self.name, self.desc)

    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            
            new_shape = self.shape
            row_maj_strides = [1 for _ in range(len(new_shape))]
            idx = len(row_maj_strides) - 2
            while idx >=0:
                row_maj_strides[idx] = row_maj_strides[idx + 1] * self.shape[idx + 1]
                idx -= 1

            new_offset = 0
            for i, key in enumerate(keys):
                if isinstance (key, slice):
                    if key.start is None:
                        key = slice(0, key.stop, key.step)
                    
                    num_rows = self.shape[i]
                    if key.stop:
                        num_rows = key.stop - key.start
                    elif key.stop is None:
                        num_rows = new_shape[i] - key.start

                    new_shape[i] = num_rows
                new_offset += key.start * row_maj_strides[i]
            new_tensor = Tensor(new_shape, data_type=self.data_type)
            SymbolTable.create_entry(new_tensor, reuse_t=self, offset=new_offset)
            return new_tensor

        else:
            return self
