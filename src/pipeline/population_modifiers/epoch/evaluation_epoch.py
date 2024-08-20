from src.pipeline.population_modifiers.epoch.evaluation import Evaluation
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier


class EvaluationEpoch(PopulationModifier):
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    def run(self, population: PopulationDTO) -> PopulationDTO:
        for network in population.population:
            self.evaluation.evaluate(network)
        
        self.log_population_metrics(population)
        
        return population

    def log_population_metrics(self, population: PopulationDTO):
        try:
            population.average_score = sum([network.score for network in population.population]) / len(population.population)
            population.min_score = min([network.score for network in population.population])
            population.max_score = max([network.score for network in population.population])
            print(f'EvaluationEpoch - Evaluation Epoch complete!')
            print(f'EvaluationEpoch - Average score: {population.average_score}')
            print(f'EvaluationEpoch - Min score: {population.min_score}')
            print(f'EvaluationEpoch - Max score: {population.max_score}')
        except:
            print('EvaluationEpoch - Error calculating population metrics')