
import math
import unittest

from src.neuralnet.sigmoid_node import SigmoidNode
from src.neuralnet.synapse import Synapse


class TestSigmoidNodeSimpleExample(unittest.TestCase):

    def test_two_input_nodes(self):
        node = SigmoidNode()
        node.bias = float(2)
        
        node.input_synapses = [
            self.createSynapse(activation=float(1), weight=float(3)),
            self.createSynapse(activation=float(0.5), weight=float(2))
        ]
        
        node.determine_activation()

        expected_activation = self.get_expected_activation()

        assert math.isclose(node.activation, expected_activation, abs_tol=0.001)


    def get_expected_activation(self):
        netInput = (float(1)*float(3)) + (float(0.5)*float(2))

        netInputWithBias = netInput + 2
        
        return self.sigmoid(netInputWithBias)
    
    
    def createSynapse(self, activation: float, weight: float):
        input_node = SigmoidNode()
        input_node.activation = activation
        output_node = SigmoidNode()

        synapse = Synapse(input_node, output_node, weight)

        return synapse


    def sigmoid(self, net_input: float):
        return 1 / (1 + math.exp(-net_input))