from src.pipeline.population_modifiers.epoch.competition import Competition
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_modifier import PopulationModifier


class CompetitionEpoch(PopulationModifier):
    def __init__(self, competition: Competition):
        self.competition = competition

    def run(self, population: PopulationDTO):
        for challenger in population.population:
            for challenged in population.population:
                self.competition.compete(challenger, challenged)
        
        return population