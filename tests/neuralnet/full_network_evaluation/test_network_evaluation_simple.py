import math
import unittest

from tests.neuralnet.full_network_evaluation.test_networks.simple_network_creator import SimpleNetworkCreator


class TestNetworkEvaluationSimple(unittest.TestCase):
    
    def test_network_two_node_evaluation(self):
        network = SimpleNetworkCreator().create()
        
        
        network.feed_forward()
        
        
        # you can see how this value is calculated in test_network_evaluation.drawio
        expected_activation = 0.843515 
        
        assert math.isclose(network.node_layers[1][0].activation, expected_activation, abs_tol=0.001)