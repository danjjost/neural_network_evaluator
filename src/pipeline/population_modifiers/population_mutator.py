from random import Random
from config import Config
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier


class PopulationMutator(PopulationModifier):
    def __init__(self, config: Config = Config()): 
        self.config = config
        self.mutation_percent: float = config.percent_mutation
        self.mutation_step: float = config.mutation_step
    
    def run(self, population: PopulationDTO) -> PopulationDTO:
        if self.config.debug:
            print(f'PopulationMutator - Mutating population of {len(population.population)} individuals.')
        
        for network in population.population:
            self.mutate_network(network)
            
        return population
            
    def mutate_network(self, network: Network):
        for layer in network.node_layers:
            for node in layer:
                if self.should_mutate():
                    node.bias += self.get_mutation_value()
        
        for layer in network.synapse_layers:
            for synapse in layer:
                if self.should_mutate():
                    synapse.weight += self.get_mutation_value()
                    
        return network
    
    def should_mutate(self) -> bool:
        return Random().random() < self.mutation_percent
    
    def get_mutation_value(self) -> float:
        random_between_negative_one_and_one = (Random().random() * 2) - 1
        return random_between_negative_one_and_one * self.mutation_step