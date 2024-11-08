# data handling
import numpy as np
from tqdm import tqdm
import yaml

# deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
# plotting
import matplotlib.pyplot as plt
# numerical computation
from scipy.integrate import odeint
from scipy.stats import qmc
# special class
from physics import *

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
params = './physics/params.yaml'
current_path = os.getcwd() # get the current path
# check if the \src folder exists in the current path. if yes, skip it  
# physics model
estimator = electrodialysis(print_info=True)
# load the base parameters
params = yaml.safe_load(open(f"{params}"))
estimator.set_params(params=params)


num_points = 10
N = np.array([30, 40, 50, 80, 100])
V = np.array([0.1, 0.3, 0.5, 0.8, 1.5])
T_tot = np.array([30, 60, 90, 120, 150])
# N = np.array([80])
# V = np.array([1])
# T_tot = np.array([120])
features = (N, V, T_tot)
target = np.array([0.027086021788797154, 0.27567417040713527, 0.6898858806380636, 0.9693585425167238, 0.9926057882571218])

assert N.shape == V.shape == T_tot.shape, 'Shapes of input arrays do not match'
method = 'solve_ivp'
result = []
for index in range(len(N)):
    SR, final_conc = objective(estimator=estimator, params=params, action=[N[index], V[index], T_tot[index]], steps=num_points, method=method)
    result.append(SR)
print(result)