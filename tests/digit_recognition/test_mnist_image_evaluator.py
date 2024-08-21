import unittest
from unittest.mock import MagicMock

from config import NetworkEvaluationMode
from src.digit_recognition.MNISTImage import MNISTImage
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator


class TestMNISTImageEvaluator(unittest.TestCase):
    def test_evaluate_image(self):
        network = self.create_mock_network(output_index=5)

        image = MagicMock(spec=MNISTImage)
        image_array = [0.0] * 10
        image.image_array = image_array

        expected_label = 5
        image.label = expected_label
        
        config = MagicMock()
        config.mode = NetworkEvaluationMode.TRAIN
        
        evaluator = MNISTImageEvaluator(config)
        
        
        evaluator.evaluate_image(network, image)

        
        network.set_input.assert_called_with(image_array)        
        network.feed_forward.assert_called()

        expected_output = [0.0] * 10
        expected_output[expected_label] = 1.0        
        network.back_propagate.assert_called_with(expected_output)
        
        
    def test_evaluate_image_does_not_backpropagate_when_testing(self):
        network = self.create_mock_network(output_index=5)

        image = MagicMock(spec=MNISTImage)
        image_array = [0.0] * 10
        image.image_array = image_array

        expected_label = 5
        image.label = expected_label
        
        config = MagicMock()
        config.mode = NetworkEvaluationMode.TEST
        
        evaluator = MNISTImageEvaluator(config)
        
        
        evaluator.evaluate_image(network, image)

        
        network.set_input.assert_called_with(image_array)        
        network.feed_forward.assert_called()

        network.back_propagate.assert_not_called()
        
        
    def test_score_increased_when_prediction_is_correct(self):
        network = self.create_mock_network(output_index=5)
        network.score = 0

        image = MagicMock(spec=MNISTImage)
        image_array = [0.0] * 10
        image.image_array = image_array

        expected_label = 5
        image.label = expected_label
        
        config = MagicMock()
        config.mode = NetworkEvaluationMode.TRAIN
        
        evaluator = MNISTImageEvaluator(config)
        
        
        evaluator.evaluate_image(network, image)

        
        self.assertEqual(1, network.score)
    
    def test_image_evaluator_returns_likely_digit(self):
        network = self.create_mock_network(output_index=5)
        network.score = 0

        image = MagicMock(spec=MNISTImage)
        image_array = [0.0] * 10
        image.image_array = image_array

        expected_label = 5
        image.label = expected_label
        
        config = MagicMock()
        config.mode = NetworkEvaluationMode.TRAIN
        
        evaluator = MNISTImageEvaluator(config)
        
        
        result = evaluator.evaluate_image(network, image)


        self.assertEqual(5, result)
        
    
    def create_mock_network(self, output_index: int):
        network = MagicMock()
        
        outputs = [0.0] * 10
        outputs[output_index] = 1.0
        
        network.get_outputs.return_value = outputs
        
        return network