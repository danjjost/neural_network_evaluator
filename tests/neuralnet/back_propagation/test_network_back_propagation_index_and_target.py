
import unittest

from tests.neuralnet.full_network_evaluation.test_networks.simple_network_creator import SimpleNetworkCreator



class TestNetworkBackPropagationSingle(unittest.TestCase):
    def test_back_propagation(self):
        typical_backprop_network = SimpleNetworkCreator().create()
        indexed_backprop_network = SimpleNetworkCreator().create()
        typical_backprop_network.feed_forward()
        indexed_backprop_network.feed_forward()
        
        typical_backprop_network.back_propagate([float('1')])
        indexed_backprop_network.back_propagate_node_index_and_target(0, float(1))
        
        typical_backprop_network.apply_gradients()
        indexed_backprop_network.apply_gradients()
        
        
        assert typical_backprop_network.node_layers[0][0].bias == indexed_backprop_network.node_layers[0][0].bias, f"typical_backprop_network.node_layers[0][0].bias was {typical_backprop_network.node_layers[0][0].bias}, expected {indexed_backprop_network.node_layers[0][0].bias}"
        assert typical_backprop_network.node_layers[0][1].bias == indexed_backprop_network.node_layers[0][1].bias, f"typical_backprop_network.node_layers[0][1].bias was {typical_backprop_network.node_layers[0][1].bias}, expected {indexed_backprop_network.node_layers[0][1].bias}"
        
        assert typical_backprop_network.node_layers[1][0].bias == indexed_backprop_network.node_layers[1][0].bias, f"typical_backprop_network.node_layers[1][0].bias was {typical_backprop_network.node_layers[1][0].bias}, expected {indexed_backprop_network.node_layers[1][0].bias}"
        
        assert typical_backprop_network.synapse_layers[0][0].weight == indexed_backprop_network.synapse_layers[0][0].weight, f"typical_backprop_network.synapse_layers[0][0].weight was {typical_backprop_network.synapse_layers[0][0].weight}, expected {indexed_backprop_network.synapse_layers[0][0].weight}"
        assert typical_backprop_network.synapse_layers[0][1].weight == indexed_backprop_network.synapse_layers[0][1].weight, f"typical_backprop_network.synapse_layers[0][1].weight was {typical_backprop_network.synapse_layers[0][1].weight}, expected {indexed_backprop_network.synapse_layers[0][1].weight}"