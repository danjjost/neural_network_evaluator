import os
from random import Random
from typing import Optional
from config import Config

import numpy as np
from PIL import Image

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
        img = Image.open(file_path) # type: ignore

        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        img_array_float = img_array.astype(np.float32) / 255.0
        
        return [x.item() for x in img_array_float.flatten()]