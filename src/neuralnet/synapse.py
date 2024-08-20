from typing import TYPE_CHECKING, Optional
import uuid

from config import Config

if TYPE_CHECKING:
    from src.neuralnet.sigmoid_node import SigmoidNode

class Synapse:
    def __init__(self, input_node: 'SigmoidNode', output_node: 'SigmoidNode', weight: float, config: Optional[Config] = None) -> None:
        input_node.output_synapses.append(self)
        
        output_node.input_synapses.append(self)

        # evaluation state
        self.gradients: list[float] = []
        
        # predefined state
        self.input_node: SigmoidNode = input_node
        self.output_node: SigmoidNode = output_node
        self.weight:float = weight or 0.0
        
        self.config = config or Config()
        
        
    def apply_gradients(self):
        self.weight += self.config.learning_rate * sum(self.gradients, 0.0)
        self.gradients.clear()
        
        
    def clear_evaluation_state(self):
        self.gradients.clear()