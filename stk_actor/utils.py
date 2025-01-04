import random
import torch
import torch.nn as nn
from collections import deque
from itertools import chain
import pandas as pd

class ReplayBuffer:
    """
    A FIFO replay buffer created by own
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, is_done):
        experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'is_done': is_done}
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        samples = random.sample(self.buffer, batch_size)

        return {key: [d[key] for d in samples] for key in samples[0].keys()}

    def size(self):
        return len(self.buffer)

def build_mlp(dims: list[int], activation, output_activation=None):
    mlp = []
    for i in range(len(dims) - 1):
        mlp.append(nn.Linear(dims[i], dims[i + 1]))
        mlp.append(activation)

    if output_activation:
        mlp[-1] = output_activation

    return nn.Sequential(*mlp)

def setup_optimizer(cfg_optimizer, *agents):
    def get_arguments(arguments):
        d = dict(arguments)
        if "classname" in d:
            del d["classname"]
        return d

    def get_class(arguments):
        from importlib import import_module

        if isinstance(arguments, dict):
            classname = arguments["classname"]
            module_path, class_name = classname.rsplit(".", 1)
            module = import_module(module_path)
            c = getattr(module, class_name)
            return c
        else:
            classname = arguments.classname
            module_path, class_name = classname.rsplit(".", 1)
            module = import_module(module_path)
            c = getattr(module, class_name)
            return c

    optimizer_args = get_arguments(cfg_optimizer)
    parameters = [
        agent.parameters() if isinstance(agent, nn.Module) else [agent]
        for agent in agents
    ]
    optimizer = get_class(cfg_optimizer)(chain(*parameters), **optimizer_args)
    return optimizer


def divide_continuous_discrete(experiences):
    return experiences

if __name__ == '__main__':
    rb = ReplayBuffer(100)
    for _ in range(100):
        rb.store(random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.choice([True, False]))

    print(rb.sample(10))

