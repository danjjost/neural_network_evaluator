
import unittest

from src.neuralnet.sigmoid_node import SigmoidNode
from src.neuralnet.synapse import Synapse
from src.neuralnet.to_dict.synapse_to_dict import SynapseToDict


class TestSynapseToDict(unittest.TestCase):
    def test_synapse_to_dict(self):
        input_node = SigmoidNode()
        output_node = SigmoidNode()
        weight = float(0.5)
        synapse = Synapse(input_node, output_node, weight)

        synapse_dict = SynapseToDict().to_dict(synapse)

        self.assertEqual(float(synapse_dict.get('w')), weight)