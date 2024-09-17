from typing import Optional
from config import Config, NetworkEvaluationMode
from src.digit_recognition.image_loader import ImageLoader, MNISTImage
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator
from src.neuralnet.network import Network
from src.pipeline.population_modifiers.epoch.evaluation import Evaluation
import logging


class MNISTEvaluation(Evaluation):
    
    def __init__(self, 
        config: Optional[Config] = None, 
        batch_loader: Optional[ImageLoader] = None,
        mnist_image_evaluator: Optional[MNISTImageEvaluator] = None
    ):
        self.config = config or Config()
        self.batch_loader = batch_loader or ImageLoader(self.config, None)
        self.mnist_image_evaluator = mnist_image_evaluator or MNISTImageEvaluator(self.config)
        self.config.learning_rate = 1
        
        
    def evaluate(self, network: Network):
        logging.info("MNISTEvaluation - Starting...")
        network.score = 0
        
        likely_digits: list[int] = []
        
        for i in range(self.config.training_batch_size):
            logging.info(f"MNISTEvaluation - Evaluating image {i + 1}/{self.config.training_batch_size}...")
            image = self.get_image()
            likely_digit = self.mnist_image_evaluator.evaluate_image(network, image)
            likely_digits.append(likely_digit)
            
        logging.info(f"MNISTEvaluation - Finished training batch of size {self.config.training_batch_size}.")
        
        self.apply_gradients_if_training(network)
        
        if self.is_guessing(likely_digits):
            network.score -= self.config.is_guessing_penalty
            
        logging.info(f'MNISTEvaluation - Network scored {network.score}/{self.config.training_batch_size}.')

    def apply_gradients_if_training(self, network: Network):
        if self.config.mode == NetworkEvaluationMode.TRAIN:
            logging.info("MNISTEvaluation - Applying Gradients")
            network.apply_gradients()

    
    def is_guessing(self, likely_digits: list[int]) -> bool:        
        most_common_digit = max(set(likely_digits), key=likely_digits.count)
        
        if likely_digits.count(most_common_digit) / len(likely_digits) > self.config.is_guessing_percent:
            return True
        
        return False
        
    
    def get_image(self) -> MNISTImage:
        if self.config.mode == NetworkEvaluationMode.TEST:
            return self.batch_loader.get_testing_image()
        else:
            return self.batch_loader.get_training_image()