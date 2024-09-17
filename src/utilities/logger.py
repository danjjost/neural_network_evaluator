from config import Config


class logging:
    def debug(self, message: str):
        if Config().debug:
            print(message)
            
    def log(self, message: str):
        print(message)