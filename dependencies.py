from random import Random

from azure_config import AzureConfig
from config import Config
from src.blob_network_converter import BlobNetworkParser
from src.blob_client import BlobClient
from src.digit_recognition.image_loader import ImageLoader
from src.digit_recognition.mnist_evaluation import MNISTEvaluation
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator
from src.neuralnet.to_dict.network_to_dict import NetworkToDict
from azure.storage.blob import BlobServiceClient, ContainerClient

class Dependencies:
    def __init__(self):
        self.config = Config()
        self.azure_config = AzureConfig()
        
        self.network_to_dict = NetworkToDict()
        
        self.random = Random()
        
        self.blob_network_parser = BlobNetworkParser(self.network_to_dict)
        
        self.image_loader = ImageLoader(self.config, self.random)
        self.mnist_image_evaluator = MNISTImageEvaluator(self.config)
        self.evaluation = MNISTEvaluation(self.config, self.image_loader, self.mnist_image_evaluator)

        input_blob_container = self.initialize_container_client(self.azure_config, self.azure_config.neural_network_input_container)
        output_blob_container = self.initialize_container_client(self.azure_config, self.azure_config.neural_network_input_container)
        
        self.input_blob_client = BlobClient(self.network_to_dict, input_blob_container)
        self.output_blob_client = BlobClient(self.network_to_dict, output_blob_container)


    def initialize_container_client(self, azure_config: AzureConfig, container_name: str) -> ContainerClient:
        connection_string = azure_config.neural_network_blob_connection_string
        
        if connection_string is None:
            raise ValueError("Neural network connection string was not found! Make sure to set it in your .env or environment variables.")
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        return blob_service_client.get_container_client(container_name)

def get_dependencies() -> Dependencies:
    return Dependencies()



