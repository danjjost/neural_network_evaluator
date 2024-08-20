import json
from typing import Optional
from src.file.file_writer import FileWriter
from src.neuralnet.to_dict.network_to_dict import NetworkDictionary, NetworkToDict
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier


class SavePopulation(PopulationModifier):
    def __init__(self, 
            path: str, 
            network_to_dict: Optional[NetworkToDict] = None, 
            file_writer: Optional[FileWriter] = None):
        
        self.path = path
        
        self.network_to_dict = network_to_dict if network_to_dict is not None else NetworkToDict() 
        self.file_writer = file_writer if file_writer is not None else FileWriter()
                
    def run(self, population: PopulationDTO) -> PopulationDTO:
        json_content = self.get_json(population)
        
        self.file_writer.save(self.path, json_content)

        print(f'Saved population of size {len(population.population)} to {self.path}.')
        return population

    def get_json(self, population: PopulationDTO) -> str:
        network_dictionaries: list[NetworkDictionary] = []
        
        for network in population.population:
            network_dictionaries.append(self.network_to_dict.to_dict(network))
        
        return json.dumps(network_dictionaries)
    
    def set_path(self, path: str) -> None:
        self.path = path