
from typing import Optional

from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier
from src.pipeline.population_modifiers.evolver.population_destroyer import PopulationDestroyer
from src.pipeline.population_modifiers.evolver.population_rebuilder import PopulationRebuilder


class PopulationEvolver(PopulationModifier):
    
    def __init__(self, 
            percent_predation: float = 0.1,
            population_destroyer: Optional[PopulationDestroyer] = None, 
            population_rebuilder: Optional[PopulationRebuilder] = None):
        self.percent_predation = percent_predation or 0.1
        
        self.population_destroyer = population_destroyer or PopulationDestroyer()
        self.population_rebuilder = population_rebuilder or PopulationRebuilder()

    
    def run(self, population: PopulationDTO) -> PopulationDTO:
        number_to_replace = int(len(population.population) * self.percent_predation)

        if number_to_replace == 0 and len(population.population) > 0:
            number_to_replace = 1
            
        print(f'PopulationEvolver - Evolving {number_to_replace} individuals from population of {len(population.population)} individuals.')

        self.population_destroyer.destroy(population, number_to_replace)
        self.population_rebuilder.rebuild(population, number_to_replace)
        
        return population
