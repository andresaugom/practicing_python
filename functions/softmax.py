import numpy as np
# %%
def softmax(z):
    exp_z = np.exp(z)
    exp_z = exp_z[:, np.newaxis]
    return exp_z/np.sum(exp_z, axis=0)