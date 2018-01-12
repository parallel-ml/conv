from channel import forward
import numpy as np

data = np.random.random_sample([256, 256, 3])
output = forward(data, 10, (3, 3))
print output.shape
