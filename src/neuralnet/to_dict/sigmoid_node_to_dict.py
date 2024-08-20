from typing import TypedDict
from src.neuralnet.sigmoid_node import SigmoidNode


class NodeDict(TypedDict):
    b: float

class SigmoidNodeToDict():
    def to_dict(self, node: SigmoidNode) -> NodeDict:
        return {
            'b': node.bias
        }
        