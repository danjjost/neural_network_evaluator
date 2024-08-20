from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier


class Pipeline():
    def __init__(self):
        self.pipeline: list[PopulationModifier] = []
    
    def add(self, population_modifier: PopulationModifier):
        self.pipeline.append(population_modifier)
    
    def run(self, population: PopulationDTO) -> PopulationDTO:
        previous_population = population
        
        for pipeline_modifier in self.pipeline:
            previous_population = pipeline_modifier.run(previous_population)
            
        return previous_population