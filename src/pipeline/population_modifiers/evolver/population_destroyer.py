from src.pipeline.population import PopulationDTO


class PopulationDestroyer():
    
    def destroy(self, population: PopulationDTO, number_to_destroy: int) -> None:
        internal_number_to_destory = number_to_destroy if number_to_destroy < len(population.population) else len(population.population)
        print(f'PopulationDestroyer - Destroying {internal_number_to_destory} individuals from population of {len(population.population)} individuals.')
        
        if number_to_destroy > len(population.population):
            print(f'PopulationDestroyer - Warning: Trying to destroy {number_to_destroy} individuals from a population of {len(population.population)}, destroying all individuals instead.')
            
        # destroy the worst individuals
        population.population = sorted(population.population, key=lambda x: x.score)
        for _ in range(internal_number_to_destory):
            population.population.pop(0)
                        