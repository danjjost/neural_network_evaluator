import unittest
from unittest.mock import MagicMock

from function_app import Dependencies, NeuralNetworkEvaluator


class TestFunctionApp(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_dependencies = MagicMock(spec=Dependencies)
        self.blob_network_converter = MagicMock()
        self.evaluation = MagicMock()
        self.blob_uploader = MagicMock()
        
        self.mock_dependencies.blob_network_converter = self.blob_network_converter
        self.mock_dependencies.evaluation = self.evaluation
        self.mock_dependencies.blob_uploader = self.blob_uploader
    
    def test_converts_blob_to_network(self):
        blob = MagicMock()
        
        NeuralNetworkEvaluator(blob, d=self.mock_dependencies)        
        
        self.blob_network_converter.convert.assert_called_with(blob)
        
    def test_evaluates_network(self):
        blob = MagicMock()
        network = self.mock_network_from_blob_conversion()
        
        NeuralNetworkEvaluator(blob, d=self.mock_dependencies)
        
        self.evaluation.evaluate.assert_called_with(network)
        
    def test_uploads_network(self):
        blob = MagicMock()
        network = self.mock_network_from_blob_conversion()
        
        NeuralNetworkEvaluator(blob, d=self.mock_dependencies)
        
        self.blob_uploader.upload_blob.assert_called_with(network, isInput=False)
        
    def mock_network_from_blob_conversion(self):
        network = MagicMock()
        
        self.blob_network_converter.convert.return_value = network
        
        return network