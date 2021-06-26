import sys
import copy
import json

import qubovert
import numpy as np
import tensorflow as tf
from gym import spaces
from baselines.ppo2 import ppo2
# from baselines.a2c import a2c
# from stable_baselines import PPO2

from qubo_nn.problems.number_partitioning import NumberPartitioning
from qubo_nn.problems.max_cut import MaxCut
from qubo_nn.problems.minimum_vertex_cover import MinimumVertexCover
from qubo_nn.problems.set_packing import SetPacking
from qubo_nn.problems.max2sat import Max2SAT
from qubo_nn.problems.set_partitioning import SetPartitioning
from qubo_nn.problems.graph_coloring import GraphColoring
from qubo_nn.problems.quadratic_assignment import QuadraticAssignment
from qubo_nn.problems.quadratic_knapsack import QuadraticKnapsack
from qubo_nn.problems.max3sat import Max3SAT
from qubo_nn.problems.tsp import TSP


def model_inference(model, obs):
    obs = tf.cast(obs.reshape(1, -1), tf.float32)
    model_action = model.step(obs)
    return model_action[0].numpy()


class Config:
    def __init__(self):
        with open('simulations.json', 'r') as f:
            self.cfg = json.load(f)

    def _update_cfg(self, base_cfg, new_cfg):
        # We support one level for now.
        for k in new_cfg.keys():
            if k == 'base_cfg' or k == 'desc':
                continue
            base_cfg[k].update(new_cfg[k])

    def get_cfg(self, cfg_id):
        if cfg_id not in self.cfg:
            raise Exception(
                'Error: Key %s does not exist in simulations.json.' % cfg_id
            )

        initial_base_cfg = self.cfg["1"]
        base_cfg = self.cfg[self.cfg[cfg_id].get('base_cfg', cfg_id)]
        # All base configs are based on config "1".
        # This enables backwards compatibility when new options are added.
        self._update_cfg(initial_base_cfg, base_cfg)

        cfg = copy.deepcopy(initial_base_cfg)
        self._update_cfg(cfg, self.cfg[cfg_id])

        cfg['cfg_id'] = cfg_id
        return cfg


class Optimizer:
    def __init__(self, cfg_id):
        self.cfg = Config().get_cfg(cfg_id)
        self.env = Env()

    def train(self):
        total_timesteps = self.cfg['ppo']['total_timesteps']

        print(self.cfg)

        kwargs = {
            "env": self.env,
            "network": 'mlp',
            "total_timesteps": total_timesteps,
            "nsteps": 1,
            "ent_coef": self.cfg['ppo']['ent_coef'],
            "lr": self.cfg['ppo']['lr'],
            "vf_coef": self.cfg['ppo']['vf_coef'],
            "max_grad_norm": self.cfg['ppo']['max_grad_norm'],
            "gamma": self.cfg['ppo']['gamma'],
            "lam": self.cfg['ppo']['lam'],
            "log_interval": self.cfg['ppo']['log_interval'],
            "nminibatches": self.cfg['ppo']['nminibatches'],
            "noptepochs": self.cfg['ppo']['noptepochs'],
            "cliprange": self.cfg['ppo']['cliprange'],
            "save_interval": self.cfg['ppo']['save_interval'],
            "num_layers": self.cfg['ppo']['num_layers'],
            "num_hidden": self.cfg['ppo']['num_hidden'],
            "evaluator": self.evaluate
        }
        model = ppo2.learn(**kwargs)

        # kwargs = {
        #     "env": self.env,
        #     "network": 'mlp',
        #     "nsteps": 1,
        #     "total_timesteps": total_timesteps,
        #     "evaluator": self.evaluate
        # }
        # model = a2c.learn(**kwargs)

        # model = PPO2('MlpPolicy', self.env, n_steps=1, verbose=1, evaluator=self.evaluate)
        # model = PPO2('MlpPolicy', self.env, n_steps=1, verbose=1, nminibatches=1)
        # model.learn(total_timesteps=100000)

    def evaluate(self, model, n_episode):
        """Run an evaluation game."""
        obs = self.env.reset()
        tot_rew = 0
        done = False
        while not done:
            action = model_inference(model, obs)
            # action = np.clip(action, 0, 1)
            print(action)
            _, reward, done, _ = self.env.step(action)
            tot_rew += reward
        print("TOTAL REWARD", tot_rew)


def bin_array(num, m=8):
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


class Env:
    def __init__(self, Q=None):
        # self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 2, 2, 2, 2])
        self.action_space = spaces.Discrete(256)
        # self.action_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        # self.action_space.shape = (8,)
        self.observation_space = spaces.Discrete(2)
        self.observation_space.shape = (1,)
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None
        self.metadata = {'render.modes': ['human']}
        self.num_envs = 1

        if Q is None:
            self.Q = np.array([
                [-3., -0., -0.,  1., -0.,  1.,  1., -0.],
                [-0., -2.,  1.,  1., -0., -0., -0., -0.],
                [-0.,  1., -3.,  1., -0., -0., -0.,  1.],
                [ 1.,  1.,  1., -3., -0., -0., -0., -0.],
                [-0., -0., -0., -0., -1.,  1., -0., -0.],
                [ 1., -0., -0., -0.,  1., -3.,  1., -0.],
                [ 1., -0., -0., -0., -0.,  1., -3.,  1.],
                [-0., -0.,  1., -0., -0., -0.,  1., -2.]
            ])

        sol = solve_qubo2(self.Q)
        print(sol.T @ self.Q @ sol)

    def step(self, action, **kwargs):
        action = action[0]
        action = bin_array(action)
        # action = np.clip(action, 0, 1)
        return np.array([1]), -action.T @ self.Q @ action, True, {}

    def reset(self, **kwargs):
        return np.array([0])

    def close(self):
        pass


def solve_qubo2(item):
    qubo_size = 8  # TODO Hardcoded!

    Q = qubovert.utils.matrix_to_qubo(item.reshape(qubo_size, qubo_size))
    sol = Q.solve_bruteforce(all_solutions=False)
    sol_ = [0 for _ in range(qubo_size)]
    for k, v in sol.items():
        sol_[k] = v
    return np.array(sol_)


cfg = {
    "problems": {
        "NP": {
            "size": 16
        },
        "MC": {
            "size": [8, 10]
        },
        "MVC": {
            "size": [16, 20]
        },
        "SP": {
            "size": [20, 16]
        },
        "M2SAT": {
            "size": [16, 20]
        },
        "SPP": {
            "size": [20, 16]
        },
        "GC": {
            "size": [4, 6],
            "n_colors": 4
        },
        "QA": {
            "size": 4
        },
        "QK": {
            "size": 10,
            "constraint": 40
        },
        "M3SAT": {
            "size": [6, 10]
        },
        "TSP": {
            "size": 4
        }
    }
}


# prob = MaxCut.gen_problems(cfg, 1, **cfg["problems"]["MC"])
# qubo = MaxCut(cfg, **prob[0]).gen_qubo_matrix()
# qubo = -qubo
# print("MC")
# print(qubo)
# print(solve_qubo2(qubo))
# x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
# print(x.T @ qubo @ x)
# import pdb; pdb.set_trace()


# x = np.array([0, 0, 0, 1, 0, 1, 0, 1]); print(x.T @ qubo @ x)


# TODO: Use PPO for training this?
# The environment is:
# Action: x
# State: always same
# Reward: QUBO value xTQx


optimizer = Optimizer(sys.argv[1])
optimizer.train()
