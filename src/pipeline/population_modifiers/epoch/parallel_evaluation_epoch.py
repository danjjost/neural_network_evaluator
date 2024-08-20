from concurrent.futures import ThreadPoolExecutor
from src.pipeline.population_modifiers.epoch.evaluation import Evaluation
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier

class ParallelEvaluationEpoch(PopulationModifier):
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    def run(self, population: PopulationDTO) -> PopulationDTO:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.evaluation.evaluate, network) for network in population.population]
            
            for future in futures:
                future.result()  
        
        self.log_population_metrics(population)
        
        return population
    

    def log_population_metrics(self, population: PopulationDTO):
        try:
            if population.population:
                population.average_score = sum([network.score for network in population.population]) / len(population.population)
                population.min_score = min([network.score for network in population.population])
                population.max_score = max([network.score for network in population.population])
                print('EvaluationEpoch - Evaluation Epoch complete!')
                print(f'EvaluationEpoch - Average score: {population.average_score}')
                print(f'EvaluationEpoch - Min score: {population.min_score}')
                print(f'EvaluationEpoch - Max score: {population.max_score}')
            else:
                print('EvaluationEpoch - Population is empty.')
        except Exception as e:
            print(f'EvaluationEpoch - Error calculating population metrics: {e}')

