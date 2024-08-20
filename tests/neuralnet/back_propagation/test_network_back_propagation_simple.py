
import math
import unittest
from config import Config
from tests.neuralnet.full_network_evaluation.test_networks.simple_network_creator import SimpleNetworkCreator


class TestNetworkBackPropagation(unittest.TestCase):
    def test_back_propagation(self):
        config = Config()
        config.learning_rate = 1
        network = SimpleNetworkCreator().create(config)
        
        network.feed_forward()
        
        targetOutputs = [float('0.25')]
        
        network.back_propagate(targetOutputs)
        network.apply_gradients()
        
        assert math.isclose(network.node_layers[1][0].bias, float('0.92166'), abs_tol=0.001), f"Expected {float('0.92166')}, got {network.node_layers[1][0].bias}"
        assert math.isclose(network.node_layers[0][0].bias, float('1.999329'), abs_tol=0.001), f"Expected {float('1.993294')}, got {network.node_layers[0][0].bias}"
        assert math.isclose(network.node_layers[0][1].bias, float('2.999495'), abs_tol=0.001), f"Expected {float('2.999495')}, got {network.node_layers[0][1].bias}"
        
        assert math.isclose(network.synapse_layers[0][0].weight, float('0.22396'), abs_tol=0.001), f"Expected {float('0.22396')}, got {network.synapse_layers[0][0].weight}"
        assert math.isclose(network.synapse_layers[0][1].weight, float('0.32294'), abs_tol=0.001), f"Expected {float('0.32294')}, got {network.synapse_layers[0][1].weight}"
        
    def test_back_propagation_multiple_gradients(self):
        config = Config()
        config.learning_rate = 1
        network = SimpleNetworkCreator().create(config)
        
        network.feed_forward()
        
        targetOutputs = [float('0.25')]
        
        # this puts two gradients on each synapse and node
        network.back_propagate(targetOutputs)
        network.back_propagate(targetOutputs)
        
        
        network.apply_gradients()
        
        
        assert math.isclose(network.node_layers[1][0].bias, float('0.84332'), abs_tol=0.001), f"Expected {float('0.84332')}, got {network.node_layers[1][0].bias}"
        assert math.isclose(network.node_layers[0][0].bias, float('1.9986588'), abs_tol=0.001), f"Expected {float('1.986588')}, got {network.node_layers[0][0].bias}"
        assert math.isclose(network.node_layers[0][1].bias, float('2.99899'), abs_tol=0.001), f"Expected {float('2.99899')}, got {network.node_layers[0][1].bias}"
        
        assert math.isclose(network.synapse_layers[0][0].weight, float('0.14792'), abs_tol=0.001), f"Expected {float('0.14792')}, got {network.synapse_layers[0][0].weight}"
        assert math.isclose(network.synapse_layers[0][1].weight, float('0.24588'), abs_tol=0.001), f"Expected {float('0.24588')}, got {network.synapse_layers[0][1].weight}"        
        