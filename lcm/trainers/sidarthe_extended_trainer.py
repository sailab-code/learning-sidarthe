from lcm.trainers.sidarthe_trainer import SidartheTrainer


class SidartheExtendedTrainer(SidartheTrainer):
    def make_initial_params(self, **kwargs):
        # default initial_params
        initial_params = {
            "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * 64,
            "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 63),
            "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 10 + [0.11] * 64,
            "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 63),
            "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * 64,
            "theta": [0.371],
            "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * 64,
            "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * 64,
            "mu": [0.017] * 22 + [0.008] * (17 + 63),
            "nu": [0.027] * 22 + [0.015] * (17 + 63),
            "tau": [0.15] * 102,
            "lambda": [0.034] * 22 + [0.08] * (17 + 63),
            "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * 64,
            "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * 64,
            "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * 64,
            "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * 64,
            "phi": [0.02] * 180,
            "chi": [0.02] * 180

        }
        _params = kwargs.get("initial_params", {})

        return self.fill_missing_params(_params, initial_params)

    def make_learning_rates(self, **kwargs):
        # default learning rates
        learning_rates = super().make_learning_rates(**kwargs)
        learning_rates["phi"] = 1e-5
        learning_rates["chi"] = 1e-5

        _params = kwargs.get("learning_rates", {})

        return self.fill_missing_params(_params, learning_rates)
