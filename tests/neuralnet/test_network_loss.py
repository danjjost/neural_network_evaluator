
import math
import unittest

from src.neuralnet.network import Network

class TestNetworkLoss(unittest.TestCase):
    
    def test_perfect_network_loss(self):
        network = Network([5])
        
        final_layer = network.node_layers[0]
        
        final_layer[0].activation = float(0.5)
        final_layer[1].activation = float(0.4)
        final_layer[2].activation = float(0.3)
        final_layer[3].activation = float(0.2)
        final_layer[4].activation = float(0.1)


        network.calculate_loss([float(0.5), float(0.4), float(0.3), float(0.2), float(0.1)])
        
        
        assert final_layer[0].loss == 0.0, f"network.loss was {network.loss}, expected 0.0"
        assert final_layer[1].loss == 0.0, f"network.loss was {network.loss}, expected 0.0"
        assert final_layer[2].loss == 0.0, f"network.loss was {network.loss}, expected 0.0"
        assert final_layer[3].loss == 0.0, f"network.loss was {network.loss}, expected 0.0"
        assert final_layer[4].loss == 0.0, f"network.loss was {network.loss}, expected 0.0"
        
        assert network.loss == 0.0, f"network.loss was {network.loss}, expected 0.0"
        
        
    def test_simple_network_loss(self):
        network = Network([2])
        
        final_layer = network.node_layers[0]
        
        final_layer[0].activation = float(0.5)
        final_layer[1].activation = float(0.4)


        network.calculate_loss([float(1), float(1)])
        
        
        # (0.5 - 1)^2 = 0.25
        assert final_layer[0].loss == 0.25, f"final_layer[0].loss was {final_layer[0].loss}, expected 0.25"
        # (0.4 - 1)^2 = 0.36
        assert math.isclose(final_layer[1].loss, 0.36), f"final_layer[1].loss was {final_layer[1].loss}, expected 0.36"
        
        # 0.25 + 0.36 = 0.61 / 2 = 0.305
        assert math.isclose(network.loss, 0.305), f"network.loss was {network.loss}, expected 0.305"
    
    
    def test_network_throws_if_loss_calculation_has_the_incorrect_number_of_expected_outputs(self):
        network = Network([2])
        
        final_layer = network.node_layers[0]
        
        final_layer[0].activation = float(0.5)
        final_layer[1].activation = float(0.4)

        with self.assertRaises(ValueError):
            network.calculate_loss([float(1), float(1), float(1)])