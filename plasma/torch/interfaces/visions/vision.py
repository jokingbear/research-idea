from ..simple_module import SimpleModule
from dataclasses import dataclass


@dataclass
class Vision(SimpleModule):
    resolution: int
    nfeatures: int
