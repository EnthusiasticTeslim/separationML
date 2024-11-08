import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
import pickle

from scipy.stats import qmc
# special class
from physics import objective   


class FCN(nn.Module):
    '''Fully connected neural network'''
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 n_layers: int, 
                 output_dim: int):
        '''Initialize the neural network
        Args:
            input_dim: number of input features
            hidden_dim: number of hidden neurons
            n_layers: number of hidden layers
            output_dim: number of output features
        '''
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fci = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        '''Forward pass of the neural network
        Args:
            variable x: input tensor of shape (n, input_size)
        Returns:
            variable out: output tensor of shape (n, output_size)
        '''
        x = F.relu(self.fc1(x))
        for fc in self.fci:
            x = F.relu(fc(x))
        x = self.fc2(x)
        return x
    
class PINN(nn.Module):
    '''Physics-informed neural network'''
    def __init__(self, 
                 estimator: callable, 
                 input_dim: int, 
                 hidden_dim: int, 
                 n_layers: int,
                 output_dim: int, 
                 params: dict,
                 model_dir: str,
                 num_points: int= 10,
                 n_samples: int= 100):
        '''Initialize the PINN
        Args:
            input_dim: number of input features
            hidden_dim: number of neurons in hidden units
            output_dim: number of output features
            n_layers: number of hidden layers
            estimator: callable function to solve the ODE
            num_points: number of points to solve the ODE
            n_samples: number of LHS samples to augment the data


        '''
        super(PINN, self).__init__()
        self.fcn = FCN(input_dim, hidden_dim, n_layers, output_dim)
        self.physics_model = estimator
        self.num_points = num_points
        self.params = params
        self.model_dir = model_dir
        self.n_samples = n_samples

        # initialize the physics model
        self.physics_model.set_params(params=params)

    def preprocess_input(self, 
                         data: tuple, 
                         first_time: bool=False):
        '''Preprocess the input data (tuple) to scale to (0, 1)'''

        (N, V, T_tot) = data
        assert N.shape == V.shape == T_tot.shape, 'Shapes of input arrays do not match'
        
        x = np.column_stack(data)

        if first_time:
            # instantiate MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            # fit and transform the data
            x_scaled = scaler.fit_transform(x)
            # save the scaler to a pickle file
            with open(f'{self.model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        else:
            # load the scaler from the pickle file
            with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            # transform the data
            x_scaled = scaler.transform(x)

        return x_scaled
    
    def forward(self, 
                data: np.array):
        '''Feedforward function for the FCN for data (experiment count * n_features)'''
        # print(f'input in forward pass: {data.shape}')
        x = torch.tensor(data, dtype=torch.float32)
        out = self.fcn(x)
        return out

    def physics(self, data: tuple):
        '''solve the ODE for multiple sets of data (N, V, T_tot) for 'num_points' points to solve & returns array of solutions
        '''
    
        result = []
        for index in range(len(data[0])):
            SR, final_conc = objective(estimator=self.physics_model, params=self.params, action=[var[index] for var in data], steps=self.num_points)
            result.append(SR)

        return torch.tensor(np.array(result), dtype=torch.float32)

    def augment(self, data: tuple):
        '''Apply Latin Hypercube Sampling to the input data (N, V, T_tot) to generate more samples'''
        # original data
        original_data = np.vstack(data).T
        # obtain the range of each parameter
        l_bound = np.array([i.min() for i in data])
        u_bound = np.array([i.max() for i in data])
        # check if the bounds are the same
        assert len(l_bound) == len(u_bound), 'Lower and upper bounds do not match'
        # apply Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=len(l_bound))
        design_space = sampler.random(self.n_samples)
        # convert the LHS samples (0, 1) to the actual range
        new_design = qmc.scale(design_space, l_bound, u_bound)
        # merge the new samples with the original data and reshuflle
        augmented_data = np.vstack((original_data, new_design))
        np.random.shuffle(augmented_data)

        return (augmented_data[:, 0], augmented_data[:, 1], augmented_data[:, 2])