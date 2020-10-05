import os
import json

def list_experiments_in_dir(experiments_path):
    exps_dirs = [exp_path for exp_path in os.listdir(experiments_path) 
        if os.path.isdir(os.path.join(experiments_path, exp_path))
    ]

    experiments = []
    for exp_dir in exps_dirs:
        final_json_path = os.path.join(experiments_path, exp_dir, "final.json")
        settings_json_path = os.path.join(experiments_path, exp_dir, "settings.json")
        exp = {}
        if os.path.exists(final_json_path):
            exp['uuid'] = exp_dir
            with open(final_json_path) as json_file:
                exp['final'] = json.load(json_file)
            with open(settings_json_path) as json_file:
                exp['settings'] = json.load(json_file)
            experiments.append(exp)

    return experiments
    