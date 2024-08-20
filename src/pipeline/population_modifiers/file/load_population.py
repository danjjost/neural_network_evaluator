import json
from typing import Optional

from src.file.file_loader import FileLoader
from src.neuralnet.to_dict.network_to_dict import NetworkDictionary, NetworkToDict
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier


class LoadPopulation(PopulationModifier):
    def __init__(self,  path: str, network_to_dict: Optional[NetworkToDict] = None, file_loader: Optional[FileLoader] = None):
        self.path = path

        self.file_loader = file_loader if file_loader is not None else FileLoader()
        self.network_to_dict = network_to_dict if network_to_dict is not None else NetworkToDict()

    def run(self, population: PopulationDTO) -> PopulationDTO:
        population.clear()

        population_json = self.file_loader.load(self.path)
        if population_json == "": 
            return population
        
        json_object_list: list[NetworkDictionary] = json.loads(population_json)

        if not isinstance(json_object_list, list): # type: ignore
            raise ValueError(f"Expected population file to be a list of networks. Got: {str(type(json_object_list))} instead.")
        
        for json_object in json_object_list:
            network = self.network_to_dict.from_dict(json_object)
            population.population.append(network)
            
        print(f'Loaded population of size {len(population.population)} from {self.path}.')
        return population
    
    def set_path(self, path: str) -> None:
        self.path = path