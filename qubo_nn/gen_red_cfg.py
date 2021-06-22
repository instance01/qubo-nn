import copy
import json


problems = ["NP", "MC", "MVC", "SP", "M2SAT", "SPP", "GC", "QA", "QK", "M3SAT", "TSP", "GI", "SGI", "MCQ"]


cfg = {
    "red2": {
        "dataset_id": "red2",
        "base_cfg": "red_base",
        "problems": {
            "n_problems": 100000,
            "problems": ["NP"]
        },
        "model": {
            # "n_epochs": 100,
            "lr": 10.0,
            "fc_sizes": [[4096], [4096]]
        }
    }
}


with open("simulations.json", "a+") as f:
    for problem in problems:
        dataset_cfg_id = "red_" + problem.lower() + "_1"
        for n in range(1, 20):
            size = 4096 * (n / 20)
            cfg_ = copy.deepcopy(cfg)
            cfg_["red2"]["model"]["fc_sizes"][0] = [int(size)]
            cfg_["red2"]["problems"]["problems"] = [problem]
            cfg_id = "red_" + problem.lower() + "_" + str(n)
            cfg_[cfg_id] = cfg_["red2"]
            cfg_[cfg_id]["dataset_id"] = dataset_cfg_id
            del cfg_["red2"]
            f.write("    " + json.dumps(cfg_)[1:-1] + ",\n")
