
import unittest
from src.neuralnet.network import Network

from tests.helpers.is_close_assert import is_close_assert # type: ignore
from tests.neuralnet.full_network_evaluation.test_networks.complex_network_creator import ComplexNetworkCreator

class TestNetworkEvaluationComplex(unittest.TestCase):
    
    def test_network_seven_node_evaluation(self):
        network = ComplexNetworkCreator().create()
                
        
        network.feed_forward()
        
        
        self.validate_layer_1_activation(network)
        self.validate_layer_2_activation(network)
        self.validate_layer_3_activation(network)
        
        
    def validate_layer_1_activation(self, network: Network):
        is_close_assert(network.node_layers[0][0].activation, float('0.9706'), abs_tol=0.001)
        is_close_assert(network.node_layers[0][1].activation, float('0.9836'), abs_tol=0.001)
        
        
    def validate_layer_2_activation(self, network: Network):
        is_close_assert(network.node_layers[1][0].activation, float('0.9642'), abs_tol=0.001)
        is_close_assert(network.node_layers[1][1].activation, float('0.8784'), abs_tol=0.001)
        is_close_assert(network.node_layers[1][2].activation, float('0.9407'), abs_tol=0.001)
    
    def validate_layer_3_activation(self, network: Network):
        is_close_assert(network.node_layers[2][0].activation, float('0.9586'), abs_tol=0.001)
        is_close_assert(network.node_layers[2][1].activation, float('0.9715'), abs_tol=0.001)