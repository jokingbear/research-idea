import plasma.functional as F


class Configs(F.BaseConfigs):

    def __init__(self):
        super().__init__()

        self.device = 1
        self.other_configs = Configs.Configs2()
    

    class Configs2(F.BaseConfigs):

        def __init__(self):
            super().__init__()

            self.device = 1
            self.others = {
                'device': 1
            }


a = Configs()