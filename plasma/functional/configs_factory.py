from .pipes import BaseConfigs
from ..meta import ObjectFactory


class ConfigsFactory(ObjectFactory[str, BaseConfigs]):

    def update(self, **configs):
        for k, cfgs in configs.items():
            if k in self:
                self[k].run(cfgs)
