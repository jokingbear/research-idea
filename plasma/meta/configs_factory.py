from ..functional.pipes import BaseConfigs
from .class_registrator import ObjectFactory


class ConfigsDict(dict[str, BaseConfigs]):

    def __init__(self, configs_factory:ObjectFactory[str, BaseConfigs]):
        super().__init__(configs_factory.shared_init())

    def update(self, **configs):
        for k, cfgs in configs.items():
            if k in self:
                self[k].run(**cfgs)
