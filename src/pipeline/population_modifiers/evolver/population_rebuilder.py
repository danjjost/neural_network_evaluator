from random import Random
from typing import Optional
from config import Config
from src.neuralnet.network import Network
from src.neuralnet.to_dict.network_to_dict import NetworkToDict
from src.pipeline.population import PopulationDTO
from src.pipeline.top_percentile_selector import TopPercentileSelector


class PopulationRebuilder():
    def __init__(self, top_percentile_selector: Optional[TopPercentileSelector] = None):
        self.top_percentile_selector = top_percentile_selector or TopPercentileSelector(Config())
                
    def rebuild(self, population: PopulationDTO, number_to_copy: int) -> PopulationDTO:
        print(f'PopulationRebuilder - Rebuilding {number_to_copy} individuals from population of {len(population.population)} individuals.')
        
        top_percentile_networks = self.top_percentile_selector.select(population)
        
        for _ in range(number_to_copy):
            network = Random().choice(top_percentile_networks)
            cloned_network = self.clone(network)
            population.population.append(cloned_network)
            
        return population
    
    def clone(self, network: Network):
        dictionary = NetworkToDict().to_dict(network) 
        new_network = NetworkToDict().from_dict(dictionary)
        return new_network