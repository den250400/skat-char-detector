import numpy as np
import random


N = 100000
FRACTION = 0.5

arrays = np.zeros((N, 4), dtype=np.int32)

# Initialize arrays
for i in range(len(arrays) - int(len(arrays)*FRACTION)):
    arrays[i, 0] = 1

for i in range(len(arrays) - int(len(arrays)*FRACTION), len(arrays)):
    arrays[i, 1] = 1

# Random cyclic shift from 0 to 3
for i in range(len(arrays)):
    arrays[i] = np.roll(arrays[i], random.randint(0, 3))

# Compute the probability distribution
probability = np.zeros(4, dtype=np.float32)
for i in range(len(probability)):
    probability[i] = len(arrays[arrays[:, i] == 1])

probability = probability / len(arrays)
print(probability)
