from typing import Optional
from uuid import uuid4
from config import Config
from src.neuralnet.sigmoid_node import SigmoidNode
from src.neuralnet.synapse import Synapse


class Network():

    def __init__(self, dimensions: list[int], config: Optional[Config] = None):
        self.id: str = str(uuid4())
        
        self.config = config
        
        self.node_layers: list[list[SigmoidNode]] = []
        self.synapse_layers: list[list[Synapse]] = []
        
        self.score: float = 0
        
        self.initialize(dimensions)


    def set_input(self, input_values: list[float]):
        if len(input_values) != len(self.node_layers[0]):
            raise ValueError(f"Expected {len(self.node_layers[0])} inputs, but got {len(input_values)}!")
        
        for node_index in range(len(input_values)):
            self.node_layers[0][node_index].starting_input = input_values[node_index]


    def initialize(self, dimensions: list[int]):
        self.initialize_node_layers(dimensions)
        self.initialize_synapse_layers(dimensions)


    def initialize_node_layers(self, dimensions: list[int]):
        for layer_index in range(len(dimensions)):
            self.node_layers.append(self.create_node_layer(dimensions[layer_index]))
            
            
    def create_node_layer(self, size: int) -> list[SigmoidNode]:
        layer: list[SigmoidNode] = []

        for _ in range(size):
            layer.append(SigmoidNode(config=self.config))

        return layer
            
            
    def initialize_synapse_layers(self, dimensions: list[int]):
        number_of_synapse_layers = len(dimensions) - 1
        
        for synapse_layer_index in range(number_of_synapse_layers):
            self.synapse_layers.append([])
            
        for synapse_layer_index in range(number_of_synapse_layers):
            current_layer_index = synapse_layer_index
            next_layer_index = synapse_layer_index + 1
            
            self.synapse_layers[current_layer_index] = self.create_synapse_layer(current_layer_index, next_layer_index)


    def create_synapse_layer(self, current_layer_index: int, next_layer_index: int) -> list[Synapse]:
        synapse_layer: list[Synapse] = []
        
        for current_node in self.node_layers[current_layer_index]:
            for next_node in self.node_layers[next_layer_index]:        
                synapse_layer.append(Synapse(current_node, next_node, 0.0, config=self.config))
                
        return synapse_layer
    
    
    def clear_evaluation_state(self):
        for layer in self.node_layers:
            for node in layer:
                node.clear_evaluation_state()
                
                
    def feed_forward(self):
        for layer in range(len(self.node_layers)):
            for node in range(len(self.node_layers[layer])):
                self.node_layers[layer][node].determine_activation()
                
                
    def get_results(self) -> list[float]:
        return [node.activation for node in self.node_layers[-1]]


    def calculate_loss(self, expected_output: list[float]):
        self.validate_for_loss_calculation(expected_output)
        
        final_layer = self.get_final_layer()
        
        self.calculate_node_losses(expected_output, final_layer)
        
        unaveraged_loss = self.calculate_unaveraged_loss(final_layer)
        self.loss: float = unaveraged_loss / float(len(final_layer))


    def calculate_node_losses(self, expected_output: list[float], final_layer: list[SigmoidNode]):
        for node_index in range(len(final_layer)):
            unsquared_loss = final_layer[node_index].activation - expected_output[node_index]
            final_layer[node_index].loss = unsquared_loss ** 2


    def calculate_unaveraged_loss(self, final_layer: list[SigmoidNode]):
        total_unaveraged_loss: float = 0.0
        
        for node in final_layer:
            total_unaveraged_loss += node.loss
            
        return total_unaveraged_loss

    
    def get_final_layer(self) -> list[SigmoidNode]:
        return self.node_layers[-1]
    
    
    def get_outputs(self) -> list[float]:
        final_layer = self.get_final_layer()
        return [node.activation for node in final_layer]    
        
        
    def validate_for_loss_calculation(self, expected_output: list[float]):
        expected_number_of_outputs = len(expected_output)
        final_layer_number_of_nodes = len(self.node_layers[-1])
        
        if expected_number_of_outputs != final_layer_number_of_nodes:
            raise ValueError(f"Expected {expected_number_of_outputs} outputs, but got {final_layer_number_of_nodes}!")
        
        for node_index, node in enumerate(self.node_layers[-1]):
            if node.activation is None: # type: ignore
                raise ValueError(f"Node at index '{node_index}' has not been activated!")


    def back_propagate(self, expected_output: list[float]):
        final_layer = self.get_final_layer()
        
        for node_index in range(len(final_layer)):
            node = final_layer[node_index]
            errorSignal = (expected_output[node_index] - node.activation)
            self.back_propagate_node(node, errorSignal)
            
    
    def back_propagate_node_index_and_target(self, index: int, target: float):
        final_layer = self.get_final_layer()
        node = final_layer[index]
        errorSignal = (target - node.activation)
        
        self.back_propagate_node(node, errorSignal)
    
    
    def back_propagate_node(self, node: SigmoidNode, errorSignal: float):
        gradient = errorSignal * node.activation * (1.0 - node.activation)
        node.gradients.append(gradient)
        
        for synapse in node.input_synapses:
            synapse.gradients.append(gradient * synapse.input_node.activation)
            self.back_propagate_node(synapse.input_node, gradient * synapse.weight)
            
        
    def apply_gradients(self):
        for layer in self.synapse_layers:
            for synapse in layer:
                synapse.apply_gradients()
                
        for layer in self.node_layers:
            for node in layer:
                node.apply_gradients()
