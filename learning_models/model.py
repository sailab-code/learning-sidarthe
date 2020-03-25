from abc import ABC, abstractmethod


class LearningModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, params):
        pass

    @abstractmethod
    def eval(self, x):
        pass

    @staticmethod
    @abstractmethod
    def loss(y, y_hat):
        pass

