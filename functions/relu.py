# %%
import numpy as np
# %%
def relu(x):
    x = np.array(x)
    return x * (x >= 0)
# %%
