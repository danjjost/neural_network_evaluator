from random import Random
import unittest
from azure_config import AzureConfig
from config import Config
from src.digit_recognition.blob_storage_image_loader import BlobStorageImageLoader


class TestBlobStorageImageLoader(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.debug = True
        self.azure_config = AzureConfig()
        self.random = Random()
        
        self.image_loader = BlobStorageImageLoader(self.config, self.azure_config, self.random)

    def test_get_training_image(self):
        image = self.image_loader.get_training_image()
        assert image is not None
        assert len(image.image_array) == 784
        
        
    def test_get_testing_image(self):
        image = self.image_loader.get_testing_image()
        assert image is not None
        assert len(image.image_array) == 784