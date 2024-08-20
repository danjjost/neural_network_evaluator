from typing import Optional
import numpy as np

from config import Config
from src.neuralnet.synapse import Synapse

class SigmoidNode():
    def __init__(self, config: Optional[Config] = None) -> None:        
        # evaluation state
        self.starting_input: Optional[float] = None
        self.activation: float = 0.0
        self.loss: float = 0.0
        self.gradients: list[float] = []
        
        # predefined state
        self.bias: float = 0.0

        self.input_synapses: list[Synapse] = []
        self.output_synapses: list[Synapse] = []
        self.config = config or Config()
        
        
    def apply_gradients(self):
        self.bias += self.config.learning_rate * sum(self.gradients)
        self.gradients.clear()


    def determine_activation(self) -> float:
        if self.config.debug: self.validate()

        net_input = self.get_net_input()

        self.activation = self.activation_function(net_input + self.bias)
        
        return self.activation


    def validate(self):
        if self.starting_input is not None and (len(self.input_synapses) > 0):
            raise ValueError(f"Node appears to be a first-layer node, but has input synapses and starting input! If this is a first-layer node, please remove the input synapses.")
        
        
        if (len(self.input_synapses) == 0) and self.starting_input is None:
            raise ValueError(f"Node appears to be a first-layer node, but has no starting input! If this is a first-layer node, please explicitly set the starting input.")


    def get_net_input(self):
        net_input = float('0')
        
        if self.starting_input is not None:
            return self.starting_input
        
        for synapse in self.input_synapses:
            net_input += synapse.input_node.activation * synapse.weight

        return net_input


    def activation_function(self, netInput: float) -> float:
        return 1.0 / (1.0 + np.exp(-netInput))
    

    def clear_evaluation_state(self) -> None:
        self.starting_input = None
        self.activation = 0.0
        self.loss = 0.0
        
    def to_dict(self):
        return {
            'bias': self.bias,
        }
        