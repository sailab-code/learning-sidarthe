from typing import Dict

from learning_models.optimizers.new_sir_optimizer import NewSirOptimizer

class TiedOptimizer(NewSirOptimizer):

    def __init__(self, params: Dict, learning_rates: Dict, momentum=True, m=None, a=None, summary=None,
                 tied_params=None):
        super().__init__(params, learning_rates, momentum, m, a, summary)
        if tied_params is None:
            tied_params = {}

        self.momentum = momentum
        self.m = m
        self.a = a
        self.summary = summary
        self.epoch = 1

        params_list = []
        for key, value in params.items():
            if key in tied_params:
                # this param was tied to another param, so we skip it
                continue

            param_dict = {
                "params": value,
                "name": key,
                "lr": learning_rates[key]
            }
            params_list.append(param_dict)

        defaults = dict()

        super(NewSirOptimizer, self).__init__(params_list, defaults)