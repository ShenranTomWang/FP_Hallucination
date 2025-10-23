from abc import ABC, abstractmethod

class Template(ABC):
    @abstractmethod
    def generate(self):
        pass