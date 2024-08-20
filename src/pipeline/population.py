import uuid
from src.neuralnet.network import Network


class PopulationDTO():
    def __init__(self, population: list[Network] = []) -> None:
        self.population: list[Network] = population
        self.average_score: float = 0
        self.min_score: float = 0
        self.max_score: float = 0
        
    def clear(self) -> None:
        self.population = []