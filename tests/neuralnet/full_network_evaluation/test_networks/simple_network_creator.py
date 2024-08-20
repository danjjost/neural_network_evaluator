
from typing import Optional
from config import Config
from src.neuralnet.network import Network


# A simple 3 node network with known values used for testing.
# The full diagram is available in test_network_evaluation.drawio
class SimpleNetworkCreator():
    def create(self, config: Optional[Config] = None):
        network = Network([2, 1], config)
        
        network.node_layers[0][0].starting_input = float('1.5')
        network.node_layers[0][0].bias = float('2')
        
        network.node_layers[0][1].starting_input = float('1.1')
        network.node_layers[0][1].bias = float('3')
        
        network.synapse_layers[0][0].weight = float('0.3')
        network.synapse_layers[0][1].weight = float('0.4')
        
        network.node_layers[1][0].bias = float('1')
        
        return network
        