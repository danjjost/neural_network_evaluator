
import unittest

from src.neuralnet.network import Network

class TestNetworkConstruction(unittest.TestCase):
    
    def test_network_sets_input_layer(self):
        network = Network([3, 1])
        
        network.set_input([float(0.1), float(0.2), float(0.3)])
        
        assert network.node_layers[0][0].starting_input == 0.1
        assert network.node_layers[0][1].starting_input == 0.2
        assert network.node_layers[0][2].starting_input == 0.3
    
    def test_network_builds_with_correct_number_of_layers(self):
        network1 = Network([1, 1, 1, 1])
        network2 = Network([1, 1])
        network3 = Network([1])

        assert len(network1.node_layers) == 4, f"network1 had {len(network1.node_layers)} layers, expected 3"
        assert len(network2.node_layers) == 2, f"network2 had {len(network2.node_layers)} layers, expected 1"
        assert len(network3.node_layers) == 1, f"network3 had {len(network3.node_layers)} layers, expected 0"
        
        
    def test_network_builds_with_correct_number_of_synapse_layers(self):
        network1 = Network([1, 1, 1, 1])
        network2 = Network([1, 1])
        network3 = Network([1])

        assert len(network1.synapse_layers) == 3, f"network1 had {len(network1.synapse_layers)} synapse layers, expected 3"
        assert len(network2.synapse_layers) == 1, f"network2 had {len(network2.synapse_layers)} synapse layers, expected 1"
        assert len(network3.synapse_layers) == 0, f"network3 had {len(network2.synapse_layers)} synapse layers, expected 0"
        
        
    def test_network_builds_with_correct_dimensions(self):
        network = Network([2, 5, 1])

        assert len(network.node_layers[0]) == 2, f"network.layers[0] had {len(network.node_layers[0])} nodes, expected 2"
        assert len(network.node_layers[1]) == 5, f"network.layers[1] had {len(network.node_layers[1])} nodes, expected 5"
        assert len(network.node_layers[2]) == 1, f"network.layers[2] had {len(network.node_layers[2])} nodes, expected 1"
        
        assert len(network.synapse_layers) == 2
        
        
    def test_network_builds_with_synapse_connections(self):
        network = Network([1, 2])
        
        network.node_layers[0][0].bias = float(1.2)
        network.node_layers[1][0].bias = float(1.3)
        network.node_layers[1][1].bias = float(1.4)
        
        
        synapse_layer = network.synapse_layers[0]
        
        assert synapse_layer[0].input_node.bias == 1.2
        assert synapse_layer[0].output_node.bias == 1.3
        
        assert synapse_layer[1].input_node.bias == 1.2
        assert synapse_layer[1].output_node.bias == 1.4
    
    
    def test_network_builds_synapse_connections_for_hidden_layers(self):
        network = Network([2, 1, 2])
        
        network.node_layers[1][0].bias = float(1.2)
        network.node_layers[2][0].bias = float(1.3)
        network.node_layers[2][1].bias = float(1.4)
        
        second_synapse_layer = network.synapse_layers[1]

        
        assert second_synapse_layer[0].input_node.bias == 1.2
        assert second_synapse_layer[0].output_node.bias == 1.3
        
        assert second_synapse_layer[1].input_node.bias == 1.2
        assert second_synapse_layer[1].output_node.bias == 1.4        


    def test_network_clears_evaluation_state(self):
        network = Network([1, 1])
        
        network.node_layers[0][0].activation = float(0.5)
        network.node_layers[0][0].starting_input = float(0.5)
        
        network.node_layers[1][0].activation = float(0.5)
        network.node_layers[1][0].starting_input = float(0.5)
        
        network.clear_evaluation_state()
        
        assert network.node_layers[0][0].activation == 0
        assert network.node_layers[0][0].starting_input == None
    
        assert network.node_layers[1][0].activation == 0
        assert network.node_layers[1][0].starting_input == None