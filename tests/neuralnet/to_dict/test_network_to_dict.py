
import json
import unittest

from src.neuralnet.network import Network
from src.neuralnet.to_dict.network_to_dict import NetworkToDict


class TestNetworkToDict(unittest.TestCase):
    def test_network_to_dict(self):
        network = Network([1,2])
        #    / O
        #  O - O
        
        network.node_layers[0][0].bias = float(1.2)
        
        network.synapse_layers[0][0].weight = float(1.3)
        network.synapse_layers[0][1].weight = float(1.4)
        
        network.node_layers[1][0].bias = float(1.5)
        network.node_layers[1][1].bias = float(1.6)
        
        
        dictionary_network = NetworkToDict().to_dict(network)
        
        
        reconstructed_network = NetworkToDict().from_dict(dictionary_network)
        
        
        self.assertEqual(reconstructed_network.node_layers[0][0].bias, float(1.2))
        
        self.assertEqual(reconstructed_network.synapse_layers[0][0].weight, float(1.3))
        self.assertEqual(reconstructed_network.synapse_layers[0][1].weight, float(1.4))
        
        self.assertEqual(reconstructed_network.node_layers[1][0].bias, float(1.5))
        self.assertEqual(reconstructed_network.node_layers[1][1].bias, float(1.6))
        
        
    def test_network_to_schema(self):
        network = Network([5, 4, 2])
     
        network_dictionary = NetworkToDict().to_dict(network)
     
        
        schema = NetworkToDict().get_network_schema(network_dictionary)
        
        
        self.assertEqual(schema, [5, 4, 2])
        
    def test_network_node_references(self):
        network = Network([1,2])
        #    / O
        #  O - O
        
        network.node_layers[0][0].bias = float(1.2)
        
        network.synapse_layers[0][1].weight = float(1.3)

        network.node_layers[1][1].bias = float(1.5)
        
        dictionary_network = NetworkToDict().to_dict(network)
        
        
        reconstructed_network = NetworkToDict().from_dict(dictionary_network)
        
        
        second_synapse = reconstructed_network.synapse_layers[0][1]
        
        self.assertEqual(second_synapse.weight, float(1.3))
        
        self.assertEqual(second_synapse.input_node.bias, float(1.2))
        
        self.assertEqual(second_synapse.output_node.bias, float(1.5))
        
    def test_dictionary_parsing_to_json(self):
        network = Network([1,2])
        #    / O
        #  O - O
        
        network.node_layers[0][0].bias = float(1.2)
        
        network.synapse_layers[0][0].weight = float(1.3)
        network.synapse_layers[0][1].weight = float(1.4)
        
        network.node_layers[1][0].bias = float(1.5)
        network.node_layers[1][1].bias = float(1.6)
        
        
        dictionary_network = NetworkToDict().to_dict(network)
        
        json_string = json.dumps(dictionary_network)
        dict_from_json = json.loads(json_string)

        
        reconstructed_betwork = NetworkToDict().from_dict(dict_from_json)
        
        
        self.assertEqual(reconstructed_betwork.node_layers[0][0].bias, float(1.2))
        
        self.assertEqual(reconstructed_betwork.synapse_layers[0][0].weight, float(1.3))
        self.assertEqual(reconstructed_betwork.synapse_layers[0][1].weight, float(1.4))
        
        self.assertEqual(reconstructed_betwork.node_layers[1][0].bias, float(1.5))
        self.assertEqual(reconstructed_betwork.node_layers[1][1].bias, float(1.6))
        
    def test_dictionary_includes_id(self):
        network = Network([5, 4, 2])
     

        network_dictionary = NetworkToDict().to_dict(network)
        reconstructedNetwork = NetworkToDict().from_dict(network_dictionary)
        
        
        assert reconstructedNetwork.id == network.id
    
    def test_dictionary_includes_score(self):
        network = Network([5, 4, 2])
        network.score = 0.5

        network_dictionary = NetworkToDict().to_dict(network)
        reconstructedNetwork = NetworkToDict().from_dict(network_dictionary)
        
        
        assert reconstructedNetwork.score == network.score