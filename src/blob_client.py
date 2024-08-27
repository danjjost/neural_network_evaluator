import json
from src.neuralnet.network import Network
from src.neuralnet.to_dict.network_to_dict import NetworkToDict
from azure.storage.blob import ContainerClient

# Converts a network to json and uploads it to a blob
class BlobClient:
    def __init__(self, network_to_dict: NetworkToDict, container: ContainerClient) -> None:
        self.network_to_dict = network_to_dict
        self.container = container


    def upload_batch (self, networks: list[Network]):
        for network in networks:
            self.upload_blob(network)        
    
    def upload_blob(self, network: Network):
        network_dictionary = self.network_to_dict.to_dict(network)
        network_json: str = json.dumps(network_dictionary)
        
        blob_client = self.container.get_blob_client(blob=network.id)
        blob_client.upload_blob(network_json, blob_type="BlockBlob", overwrite=True) # type: ignore
