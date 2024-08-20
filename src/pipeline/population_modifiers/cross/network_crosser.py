from random import Random
from src.neuralnet.network import Network

class NetworkCrosser:
    def cross(self, network: Network, donor: Network):
        if self.get_random_boolean():
            self.cross_synapse(network, donor)
        else:
            self.cross_node(network, donor)
        
        
    def get_random_boolean(self) -> bool:
        return Random().random() > 0.5
    
    
    def cross_synapse(self, network: Network, donor: Network):
        random_synapse_layer_index = Random().randint(0, len(network.synapse_layers) - 1)
        random_synapse_index = Random().randint(0, len(network.synapse_layers[random_synapse_layer_index]) - 1)
        
        network.synapse_layers[random_synapse_layer_index][random_synapse_index] = donor.synapse_layers[random_synapse_layer_index][random_synapse_index]
        
        
    def cross_node(self, network: Network, donor: Network):
        random_node_layer_index = Random().randint(0, len(network.node_layers) - 1)
        random_node_index = Random().randint(0, len(network.node_layers[random_node_layer_index]) - 1)
        
        network.node_layers[random_node_layer_index][random_node_index] = donor.node_layers[random_node_layer_index][random_node_index]    
    