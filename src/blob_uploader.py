import json
from azure_config import AzureConfig
from config import Config
from src.neuralnet.network import Network
from src.neuralnet.to_dict.network_to_dict import NetworkToDict
from azure.storage.blob import BlobServiceClient, ContainerClient

# Converts a network to json and uploads it to a blob
class BlobUploader:
    def __init__(self, config: Config, azure_config: AzureConfig, network_to_dict: NetworkToDict) -> None:
        self.config = config
        self.azure_config: AzureConfig = azure_config
        
        self.network_to_dict = network_to_dict
        
        self.input_container_client = self.initialize_container_client(azure_config, azure_config.neural_network_input_container)
        self.output_container_client = self.initialize_container_client(azure_config, azure_config.neural_network_output_container)

    def initialize_container_client(self, azure_config: AzureConfig, container_name: str) -> ContainerClient:
        connection_string = azure_config.neural_network_blob_connection_string
        
        if connection_string is None:
            raise ValueError("Neural network connection string was not found! Make sure to set it in your .env or environment variables.")
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        return blob_service_client.get_container_client(container_name)
        
    
    def upload_blob(self, network: Network, isInput: bool = True):
        network_dictionary = self.network_to_dict.to_dict(network)
        network_json: str = json.dumps(network_dictionary)
        
        if isInput:
            blob_client = self.input_container_client.get_blob_client(blob=network.id)
            blob_client.upload_blob(network_json, blob_type="BlockBlob", overwrite=True) # type: ignore
        else:
            blob_client = self.output_container_client.get_blob_client(blob=network.id)
            blob_client.upload_blob(network_json, blob_type="BlockBlob", overwrite=True) # type: ignore


    def upload_batch (self, networks: list[Network]):
        for network in networks:
            self.upload_blob(network)