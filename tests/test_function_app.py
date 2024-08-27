import unittest
from unittest.mock import MagicMock

from function_app import NeuralNetworkEvaluator
from tests.mock_dependencies import get_mock_dependencies


class TestFunctionApp(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_dependencies = get_mock_dependencies()

        self.blob_network_parser = self.mock_dependencies.blob_network_parser
        self.evaluation = self.mock_dependencies.evaluation
        self.blob_uploader = self.mock_dependencies.output_blob_client

    def test_converts_blob_to_network(self):
        blob = MagicMock()
        
        NeuralNetworkEvaluator(blob, d=self.mock_dependencies)        
        
        self.blob_network_parser.parse.assert_called_with(blob) #type: ignore
        
    def test_evaluates_network(self):
        blob = MagicMock()
        network = self.mock_network_from_blob_conversion()
        
        NeuralNetworkEvaluator(blob, d=self.mock_dependencies)
        
        self.evaluation.evaluate.assert_called_with(network) # type: ignore
        
    def test_uploads_network(self):
        blob = MagicMock()
        network = self.mock_network_from_blob_conversion()
        
        NeuralNetworkEvaluator(blob, d=self.mock_dependencies)
        
        self.blob_uploader.upload_blob.assert_called_with(network) # type: ignore
        
    def mock_network_from_blob_conversion(self):
        network = MagicMock()
        
        self.blob_network_parser.parse.return_value = network # type: ignore
        
        return network