
import os
from random import Random
from typing import Optional
from config import Config

from src.digit_recognition.MNISTImage import MNISTImage
        
class ImageLoader():
    def __init__(self, config: Optional[Config], random: Optional[Random]):
        self.config = config or Config()
        self.random = random or Random()
    
    
    def get_training_image(self) -> MNISTImage:
        
        path = self.config.mnist_training_folder
        
        image = self.get_random_image(path)
        
        return image
    
    
    def get_testing_image(self) -> MNISTImage:
        path = self.config.mnist_testing_folder
        
        image = self.get_random_image(path)
        
        return image
        
        
    def get_random_image(self, container_name: str) -> MNISTImage:
        """
        Returns a random image from the specified folder.
        The folder structure should be:
        
        - base_path
        - 0
            - some_0_image.jpg
            - another_0_image.jpg
        - 1
            - some_1_image.jpg
            - another_1_image.jpg
        """
        
        random_digit = self.random.randint(0, 9)
        path = container_name + str(random_digit)
        
        all_files_in_folder = os.listdir(path)
        random_file = self.random.choice(all_files_in_folder)
        
        file_path = path + "/" + random_file
        
        image_array = self.load_image(file_path)
        
        return MNISTImage(image_array, random_digit)
    
    
    def load_image(self, file_path: str) -> list[float]:
        raise NotImplementedError("load_image must be implemented by the subclass.")