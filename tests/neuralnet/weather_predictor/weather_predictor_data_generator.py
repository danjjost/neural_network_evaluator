

import random

from tests.neuralnet.weather_predictor.weather_conditions import WeatherConditions


class WeatherPredictorDataGenerator():
    
    def __init__(self):
        self.temperature_cutoff = 0.32 # If the temperature is below this value, it is considered too cold for rain
        self.cloud_cover_cutoff = 0.5 # If the cloud cover is above this value and the temperature is above the cutoff, it is considered rainy
    
    def generate_rainy_data(self, num_samples: int) -> list[WeatherConditions]:
        rainy_data: list[WeatherConditions] = []
        
        for _ in range(num_samples):
            temperature = random.uniform(self.temperature_cutoff, 1)
            cloud_cover = random.uniform(self.cloud_cover_cutoff, 1)


            rainy_data.append(WeatherConditions(float(temperature), float(cloud_cover)))
        
        return rainy_data
       
        
    def generate_sunny_data(self, num_samples: int) -> list[WeatherConditions]:
        sunny_data: list[WeatherConditions] = []
        
        
        for _ in range(num_samples):
            temperature = random.uniform(0, self.temperature_cutoff)
            cloud_cover = random.uniform(0, self.cloud_cover_cutoff)
            
            sunny_data.append(WeatherConditions(float(temperature), float(cloud_cover)))
        
        return sunny_data