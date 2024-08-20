from enum import Enum, auto


class NetworkEvaluationMode(Enum):
    TRAIN = auto()
    TEST = auto()

class Config():
    def __init__(self):
        self.mode: NetworkEvaluationMode = NetworkEvaluationMode.TRAIN
        
        self.mnist_testing_folder: str = "./MNIST/testing/"
        self.mnist_training_folder: str = "./MNIST/training/"

        # The learning rate of the network, affects the speed of gradient descent
        self.learning_rate: float = 0.1 # Default: 0.001 
        
        # The number of training batches to run before evaluating the network and applying gradients
        self.training_batch_size: int = 32
                
        self.debug: bool = False
        
        # If a network guesses a single value more than the is_guessing_percent of the time, 
        # it will be penalized by is_guessing_penalty
        self.is_guessing_percent: float = 0.90
        self.is_guessing_penalty: float = 10