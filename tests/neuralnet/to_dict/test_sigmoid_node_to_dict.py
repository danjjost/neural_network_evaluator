
import unittest

from src.neuralnet.sigmoid_node import SigmoidNode
from src.neuralnet.to_dict.sigmoid_node_to_dict import SigmoidNodeToDict

class TestSigmoidNodeToDict(unittest.TestCase):
    def test_to_dict_returns_training_state(self):
        sigmoid_node = SigmoidNode()
        sigmoid_node.bias = float(0.5)
        
        sigmoid_node.activation = float(0.2)
        sigmoid_node.loss = float(0.1)
        sigmoid_node.gradients = [float(0.1), float(0.2)]
        
        dict_representation = SigmoidNodeToDict().to_dict(sigmoid_node)
        
        assert float(dict_representation['b']) == float(0.5) # type: ignore
        
        # should have no representation for the other fields.
        assert 'a' not in dict_representation
        assert 'l' not in dict_representation
        assert 'g' not in dict_representation