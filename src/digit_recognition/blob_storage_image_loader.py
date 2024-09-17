from random import Random
from typing import Optional
from azure_config import AzureConfig
import json

from azure.storage.blob import BlobServiceClient

from config import Config
from src.digit_recognition.MNISTImage import MNISTImage
from src.digit_recognition.image_loader import ImageLoader

class BlobStorageImageLoader(ImageLoader):
    def __init__(self, config: Optional[Config], azure_config: Optional[AzureConfig], random: Optional[Random]):
        self.config = config or Config()
        self.azure_config = azure_config or AzureConfig()
        self.random = random or Random()
        
        
        self.validate_azure_config()
        self.blob_service_client = BlobServiceClient.from_connection_string(self.azure_config.training_data_blob_connection_string) # type: ignore
    
    def validate_azure_config(self):
        if not self.azure_config:
            raise ValueError("AzureConfig is not set.")
        
        if not self.azure_config.training_data_blob_connection_string:
            raise ValueError("The training data blob connection string is not set.")
        
        if not self.azure_config.mnist_testing_container:
            raise ValueError("The testing container name is not set.")
        
        if not self.azure_config.mnist_training_container:
            raise ValueError("The training container name is not set.")
    
    def get_training_image(self) -> MNISTImage:
        containerName = self.azure_config.mnist_training_container
        image = self.get_random_image(containerName)
        return image
    
    def get_testing_image(self) -> MNISTImage:
        container_name = self.azure_config.mnist_testing_container
        image = self.get_random_image(container_name)
        return image
        
    def get_random_image(self, container_name: str) -> MNISTImage:
        """
        Returns a random image from the specified folder in the blob storage.
        The folder structure should be:
        
        - blob container (training/testing)
        - 0
            - some_0_image.jpg
            - another_0_image.jpg
        - 1
            - some_1_image.jpg
            - another_1_image.jpg
        """
        
        random_digit = self.random.randint(0, 9)
        folder = f"{random_digit}/"
        
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs(name_starts_with=folder)
        blobs = [blob.name for blob in blob_list]
        
        random_blob_name = self.random.choice(blobs)
        blob_client = container_client.get_blob_client(random_blob_name)
        download_stream = blob_client.download_blob() # type:ignore
        image_data = download_stream.readall()
        
        image_array = self.load_data_from_json(image_data)
        
        if self.config.debug:
            print (f"Loaded image from {container_name}/{random_blob_name}")
            print (image_array)
            
        return MNISTImage(image_array, random_digit)
    
    def load_data_from_json(self, json_data: bytes) -> list[float]:
        json_str = json_data.decode('utf-8')
        data = json.loads(json_str)
        return data


