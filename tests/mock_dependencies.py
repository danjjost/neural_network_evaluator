from random import Random
from unittest.mock import MagicMock
from azure_config import AzureConfig
from config import Config
from dependencies import Dependencies
from src.blob_client import BlobClient
from src.blob_network_converter import BlobNetworkParser
from src.digit_recognition.image_loader import ImageLoader
from src.digit_recognition.mnist_evaluation import MNISTEvaluation
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator
from src.neuralnet.to_dict.network_to_dict import NetworkToDict

def get_mock_dependencies() -> Dependencies:
    dependencies = MagicMock(spec=Dependencies)
    
    dependencies.config = MagicMock(spec=Config)
    dependencies.azure_config = MagicMock(spec=AzureConfig)
    
    dependencies.network_to_dict = MagicMock(spec=NetworkToDict)
    
    dependencies.random = MagicMock(spec=Random)
    
    dependencies.blob_network_parser = MagicMock(spec=BlobNetworkParser)
    
    dependencies.image_loader = MagicMock(spec=ImageLoader)
    dependencies.mnist_image_evaluator = MagicMock(spec=MNISTImageEvaluator)
    dependencies.evaluation = MagicMock(spec=MNISTEvaluation)
    
    dependencies.input_blob_client = MagicMock(spec=BlobClient)
    dependencies.output_blob_client = MagicMock(spec=BlobClient)
    
    return dependencies