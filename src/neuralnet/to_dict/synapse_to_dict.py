from typing import TypedDict
from src.neuralnet.synapse import Synapse

class SynapseDict(TypedDict):
    w: float

class SynapseToDict():
    def to_dict(self, synapse: Synapse) -> SynapseDict:
        return {
            'w': synapse.weight
        }