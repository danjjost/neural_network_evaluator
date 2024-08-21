import json
from src.neuralnet.network import Network
import azure.functions as func

from src.neuralnet.to_dict.network_to_dict import NetworkToDict


class BlobNetworkConverter:
    def __init__(self, network_to_dict: NetworkToDict) -> None:
        self.network_to_dict = network_to_dict
    
    def convert(self, blob: func.InputStream) -> Network:
        blob_string = blob.read().decode('utf-8')
        
        network_dictionary = json.loads(blob_string)
        
        return self.network_to_dict.from_dict(network_dictionary)