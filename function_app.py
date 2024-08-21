from random import Random
import azure.functions as func

from azure_config import AzureConfig
from config import Config
from src.blob_network_converter import BlobNetworkConverter
from src.blob_uploader import BlobUploader
from src.digit_recognition.image_loader import ImageLoader
from src.digit_recognition.mnist_evaluation import MNISTEvaluation
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator
from src.neuralnet.to_dict.network_to_dict import NetworkToDict

app = func.FunctionApp()

class Dependencies:
    def __init__(self):
        self.config = Config()
        self.azure_config = AzureConfig()
        self.network_to_dict = NetworkToDict()
        
        self.random = Random()
        
        self.blob_network_converter = BlobNetworkConverter(self.network_to_dict)
        self.blob_uploader = BlobUploader(self.config, self.azure_config, NetworkToDict())
        
        
        self.image_loader = ImageLoader(self.config, self.random)
        self.mnist_image_evaluator = MNISTImageEvaluator(self.config)
        self.evaluation = MNISTEvaluation(self.config, self.image_loader, self.mnist_image_evaluator)
        
def get_dependencies() -> Dependencies:
    return Dependencies()

@app.blob_trigger(arg_name="myblob", path="input", connection="AzureWebJobsStorage") # type: ignore
def BlobTriggerFunction(blob: func.InputStream, d: Dependencies = get_dependencies()):
    NeuralNetworkEvaluator(blob, d)

def NeuralNetworkEvaluator(blob: func.InputStream, d: Dependencies):
    network = d.blob_network_converter.convert(blob)
    
    d.evaluation.evaluate(network)
    
    d.blob_uploader.upload_blob(network, isInput=False)