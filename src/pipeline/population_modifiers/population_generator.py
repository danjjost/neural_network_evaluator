
import random
from typing import Optional
from config import Config
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier


class PopulationGenerator(PopulationModifier):
    def generate(self, count: int, schema: list[int], config: Optional[Config] = None) -> PopulationDTO:
        networks: list[Network] = []
        
        self.config = config or Config()
        
        for index in range(count):
            print(f'Generating network {index}/{count}')
            network = Network(schema, config=config)
            self.randomize(network)
            networks.append(network)
            
        print(f'Generated {count} networks.')
        return PopulationDTO(networks)
            
            
    def randomize(self, network: Network):
        for node_layer in network.node_layers:
            for node in node_layer:
                node.bias = self.random()
                
        for synapse_layer in network.synapse_layers:
            for synapse in synapse_layer:
                synapse.weight = self.random()
        
        return network

    def random(self) -> float:
        return random.random() * self.config.initialization_scale