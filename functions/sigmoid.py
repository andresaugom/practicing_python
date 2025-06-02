# %%
import numpy as np
# %%
def sigmoid(t):
    t = np.array(t)
    t_exp = np.exp(-t)
    return 1/(1 + t_exp)