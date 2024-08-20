
from src.neuralnet.network import Network


# A more complicated neural network with known values used for testing. 
# The full diagram is available in test_network_evaluation.drawio
class ComplexNetworkCreator():

    def create(self) -> Network:
        network = Network([2, 3, 2])
        
        self.set_node_layer_1(network)
        self.set_synapse_layer_1(network)
        
        self.set_node_layer_2(network)
        self.set_synapse_layer_2(network)
        
        self.set_node_layer_3(network)
        
        return network

    
    def set_node_layer_1(self, network: Network):
        network.node_layers[0][0].starting_input = float('1.5')
        network.node_layers[0][0].bias = float('2')

        network.node_layers[0][1].starting_input = float('1.1')
        network.node_layers[0][1].bias = float('3')
    
        
    def set_synapse_layer_1(self, network: Network):
        network.synapse_layers[0][0].weight = float('0.1')
        network.synapse_layers[0][1].weight = float('0.4')
        network.synapse_layers[0][2].weight = float('0.5')
        
        network.synapse_layers[0][3].weight = float('0.2')
        network.synapse_layers[0][4].weight = float('0.6')
        network.synapse_layers[0][5].weight = float('1.2')
        
    
    def set_node_layer_2(self, network: Network):
        network.node_layers[1][0].bias = float('3')
        network.node_layers[1][1].bias = float('1')
        network.node_layers[1][2].bias = float('1.1')
        
        
    def set_synapse_layer_2(self, network: Network):
        network.synapse_layers[1][0].weight = float('1.2')
        network.synapse_layers[1][1].weight = float('1.4')
        
        network.synapse_layers[1][2].weight = float('0.2')
        network.synapse_layers[1][3].weight = float('0.4')
        
        network.synapse_layers[1][4].weight = float('1.5')
        network.synapse_layers[1][5].weight = float('1.3')
        
        
        
    def set_node_layer_3(self, network: Network):
        network.node_layers[2][0].bias = float('0.4')
        network.node_layers[2][1].bias = float('0.6')