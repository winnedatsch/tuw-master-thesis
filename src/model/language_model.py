from abc import ABC, abstractmethod

class LanguageModel(ABC):
    @abstractmethod
    def score(self, texts):
        pass