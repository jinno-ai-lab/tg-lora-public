import random

import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_reproducibility():
    set_seed(42)
    a = random.random()
    b = np.random.rand()
    c = torch.rand(3)

    set_seed(42)
    d = random.random()
    e = np.random.rand()
    f = torch.rand(3)

    assert a == d
    assert b == e
    assert torch.allclose(c, f)


def test_set_seed_different_seeds_differ():
    set_seed(42)
    a = random.random()

    set_seed(123)
    b = random.random()

    assert a != b
