import numpy as np
import torch


data = np.random.uniform(0, 1, (3,2))

print(data)

print(np.mean(data, axis=1))

