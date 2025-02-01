import numpy as np

from ....functional import AutoPipe
from abc import abstractmethod


class Preprocessor(AutoPipe):

    @abstractmethod    
    def run(self, image:np.ndarray) -> np.ndarray:
        pass
