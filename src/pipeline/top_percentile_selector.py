from config import Config
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO


class TopPercentileSelector():
    def __init__(self, config: Config):
        self.percentile = config.top_percentile
        
    def select(self, population: PopulationDTO) -> list[Network]:
        population.population.sort(key=lambda network: network.score, reverse=True)
        
        top_percentile = population.population[:int(len(population.population) * self.percentile)]
        
        return top_percentile