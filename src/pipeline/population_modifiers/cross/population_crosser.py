from random import Random
from typing import Optional
from config import Config
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.cross.network_crosser import NetworkCrosser
from src.pipeline.population_modifiers.population_modifier import PopulationModifier
from src.pipeline.top_percentile_selector import TopPercentileSelector
from src.utilities.number_of_crosses_calculator import NumberOfCrossesCalculator


class PopulationCrosser(PopulationModifier):
    def __init__(self, 
                 config: Config, 
                 network_crosser: NetworkCrosser = NetworkCrosser(), 
                 number_of_crosses_calculator: Optional[NumberOfCrossesCalculator] = None, 
                 top_percentile_selector: Optional[TopPercentileSelector] = None):
        
        self.cross_percent = config.cross_percent
        self.number_of_crosses_calculator = number_of_crosses_calculator or NumberOfCrossesCalculator(config)
        self.network_crosser = network_crosser
        self.top_percentile_selector = top_percentile_selector or TopPercentileSelector(config)
        
    def run(self, population: PopulationDTO) -> PopulationDTO:
        print("Population Crosser - Running...")
        number_of_crosses = self.number_of_crosses_calculator.get_number_of_crosses(population.population[0])
        
        top_percentile = self.top_percentile_selector.select(population)
        
        for network in population.population:
            for _ in range(number_of_crosses):
                self.network_crosser.cross(network, Random().choice(top_percentile))

        return population