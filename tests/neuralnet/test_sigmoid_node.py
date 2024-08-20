
import math
import unittest

from config import Config
from src.neuralnet.sigmoid_node import SigmoidNode
from src.neuralnet.synapse import Synapse


class TestSigmoidNode(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.debug = True


    def test_first_layer_throws_without_starting_input_or_synapses(self):
        node = SigmoidNode(self.config)
        
        node.starting_input = None
        node.input_synapses = []

        with self.assertRaises(ValueError):
            node.determine_activation()


    def test_first_layer_throws_if_starting_input_and_synapses_are_present(self):
        node = SigmoidNode(self.config)
        
        node.starting_input = float(1)
        node.input_synapses = [Synapse(SigmoidNode(), node, float(1))]

        with self.assertRaises(ValueError):
            node.determine_activation()


    def test_first_layer_uses_starting_input_if_no_input_synapses_are_present(self):
        node = SigmoidNode(self.config)
        
        node.starting_input = float(0.4)
        node.bias = float(1)
        
        node.input_synapses = []
        
        node.determine_activation()
        
        assert math.isclose(node.activation, self.sigmoid(node.starting_input + node.bias), abs_tol=0.001)

    def test_clear_evaluation_state(self):
        node = SigmoidNode(self.config)
        
        node.activation = float(1)
        node.starting_input = float(1)
        node.loss = float(1)
        
        node.clear_evaluation_state()

        assert node.activation == 0
        assert node.loss == 0

        assert node.starting_input == None


    def test_throws_if_previous_node_is_not_active(self):
        current_node = SigmoidNode(self.config)
        
        previous_node = SigmoidNode(self.config)
        
        synapse = Synapse(previous_node, current_node, float(0))
        
        current_node.input_synapses = [synapse]
        previous_node.activation = None  # type: ignore

        with self.assertRaises(TypeError):
            current_node.determine_activation()


    def sigmoid(self, x: float) -> float:
            eulers_constant = float('2.7182818284590452353602874713527')
            
            return float(1) / (float(1) + (eulers_constant ** (-x)))
