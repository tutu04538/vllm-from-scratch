"""模型流水线阶段之间传递的张量。"""

from dataclasses import dataclass
import torch


@dataclass
class IntermediateTensors:
    tensors: dict[str, torch.Tensor]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)({name: value[key] for name, value in self.tensors.items()})
        return self.tensors[key]

    def __setitem__(self, key, value):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other):
        return (isinstance(other, type(self)) and self.tensors.keys() == other.tensors.keys()
                and all(torch.equal(value, other.tensors[key]) for key, value in self.items()))

    @staticmethod
    def empty_like(other):
        return IntermediateTensors({key: torch.empty_like(value) for key, value in other.items()})
