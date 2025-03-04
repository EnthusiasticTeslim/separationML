# Author: Teslim Olayiwola
# Date: 2024-12-10
# Description: Custom environment metrics for the electrodialysis simulation

import yaml
import numpy as np
import pandas as pd
from ..physics.device import NumericalModel
from ..physics.utils import simulate

# this class get information from the physics model "NumericalModel"
def rescale_action(
                    action, 
                   lb=[15, 0.1, 45, 0.1, 0.01],
                   ub=[80, 2.0, 120, 2.0, 0.5]
                   ):
    """ Rescale the action from [-1, 1] to [lb, ub] 
        Parameters
        ----------
            action: list, list of actions
            lb: list, lower bound of the action space
            ub: list, upper bound of the action space
        Returns
        -------
            new_action: list, rescaled action 
        References
        ----------
            https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    """
    assert len(action) == len(lb) == len(ub), 'size of action, lb & ub must be the same'

    new_action = action.copy()
    
    for i in range(len(new_action)):
        new_action[i] = ((new_action[i] + 1) * (ub[i] - lb[i]) / 2) + lb[i]
        if i == 0: # first variable (N) should be integer
            new_action[i] = int(new_action[i])    

    return new_action

# lb = [0, 0.3, 30] #, 1000, 0.1, 0.01] # N, Estack, T_tot, Feed, VT_dil, VT_conc
# up = [79, 2.0, 120] #, 4000, 2.0, 0.5] # N, Estack, T_tot, Feed, VT_dil, VT_conc

class RLperformanceMetrics:
    '''Custom environment metrics for the electrodialysis simulation
    Parameters:
    ----------
        print_info: bool, print the information
        params_files: str, parameters file
        steps: int, number of steps
        method: str, method to solve the physics model
    '''
    def __init__(self, 
                 print_info=False, 
                 params_files='params.yaml',
                 timesteps=20, 
                 method='solve_ivp',
                 lb: list = [15, 0.1, 45, 0.1, 0.01],
                    ub: list = [80, 2.0, 120, 2.0, 0.5]):
        self.system = NumericalModel(print_info=print_info)
        self.params = yaml.safe_load(open(params_files))
        self.timesteps = timesteps
        self.method = method
        self.lb = lb
        self.ub = ub


    def operate(self, 
                action: np.array,
                rescale: bool = True):

        '''Execute the physics model
        Parameters:
        ----------
            action: np.array, [N, Estack, T_tot, Feed, VT_dil, VT_conc]
            rescale: bool, rescale the action space
            lb: list, lower bound of the action space
            ub: list, upper bound of the action space
        Returns:
        -------
            metrics: tuple
            SR: float, separation efficiency
            EC: float, energy consumption
            Cdil_f: float, final concentration in diluate stream
            Cconc_f: float, final concentration in concentrate stream
        '''

        # extract the action
        if rescale:
            N, Estack, T_tot, VT_dil, VT_conc = rescale_action(action, lb=self.lb, ub=self.ub) # N, Estack, T_tot, Feed, VT_dil, VT_conc = action
        else:
            N, Estack, T_tot, VT_dil, VT_conc = action
        #print(f'Action: {N, Estack, T_tot, VT_dil, VT_conc}')
        # copy self.params
        params = self.params.copy()
        # update the parameters
        params['N'] = N
        params['Estack'] = Estack
        params['T_tot'] = T_tot
        params['VT_dil'] = VT_dil
        params['VT_conc'] = VT_conc
        # add the parameters to model
        self.system.set_params(params)
        # solve the physics model; herein errors such as "RuntimeWarning: invalid value encountered in sqrt" may occur
        invalid = False
        try:
            # print('Solving the physics model')
            # print(N, Estack, T_tot, VT_dil, VT_conc)
            result = simulate(model=self.system, method=self.method, dt=self.timesteps) # (model, dt: int = 10, method: str = 'solve_ivp'
            # print(result)
            # print('Physics model solved')
        except Exception as e:
            # print(f'Error: {e}')
            invalid = True # set invalid to True if there is an error
            result = {'SR': 0, 'EC': 10, 'Cdil_f': 1000, 'Cconc_f': 1000}
        # extract the results
        SR = result['SR'] 
        EC = result['EC'] 
        Cdil_f = result['Cdil_f'] 
        Cconc_f = result['Cconc_f']
        
        return SR, EC, Cdil_f, Cconc_f, invalid


def reward_one(SR, EC=None, weight= None):
    '''Objective function to maximize SR and minimize EC
    Parameters:
    ----------
        SR: float, separation efficiency
    Returns:
    -------
        reward: float
    '''

    return SR
    
def reward_two(SR, EC, weight=[1, -1]):
    '''Objective function to maximize SR and minimize EC
    Parameters:
    ----------
    SR: float, separation efficiency
    EC: float, energy consumption

    Returns:
    -------
    reward: float
    '''
    alpha, beta = weight
    # max SR is 100% and min EC is 0
    return alpha * SR + beta * (EC/10)

def reward_three(SR, EC, weight=[1, 1]):
    '''Objective function to maximize SR and minimize EC
    Parameters:
    ----------
    SR: float, separation efficiency
    EC: float, energy consumption

    Returns:
    -------
    reward: float
    '''
    alpha, beta = weight
    # max SR is 100% and min EC is 0
    return alpha * SR + beta * (1 - (EC/10))
    
def rescale_eposide_rewards(original_rewards, length):
    """ Rescale the rewards by the length of the episode (if the episode is truncated)
    Parameters
    ----------
        original_rewards: list, list of rewards
        length: list, length of the episode
    Returns
    -------
        rewards: list, rescaled rewards
    """
    assert len(original_rewards) == len(length), 'Length of rewards and length should be the same'

    # copy the rewards
    rewards = original_rewards.copy()
    # rescale
    for i in range(len(rewards)):
        rewards[i] = rewards[i] / length[i]

    return rewards

def test_controller(env, 
                    model, n_steps=10,
                    rescale_action=rescale_action, 
                    ub=[80, 2.0, 120, 2.0, 0.5], lb=[15, 0.1, 45, 0.1, 0.01],
                    id='ppo', reward_id=1, write_path='data'):
    """Test the controller
    Parameters
    ----------
        env: gym environment
        model: trained model
        n_steps: int, number of steps
        rescale_action: function, rescale the action
        id: str, id of the model
        reward_id: int, id of the reward function
    Returns
    -------
        df: pandas dataframe, dataframe of the results
    """
    action_recorder = []
    obs_recorder = []
    """Test the controller"""
    for step in range(n_steps):
        obs, info = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"Step = {step + 1}")
        print(f"Action = {rescale_action(action, ub=ub, lb=lb)}")
        action_recorder.append(rescale_action(action, ub=ub, lb=lb))
        obs, reward, terminated, truncated, info = env.step(action) # state, reward, terminated, truncated, info 
        obs_recorder.append(obs)
        print(f"obs = {obs}\nreward = {reward}\nterminated ={terminated}")
        if terminated:
            print(f"Goal reached! Reward: {reward}")
            break

    # append the action_recorder and observations
    action_recorder = np.array(action_recorder).astype(np.float32) # N, Estack, T_tot, VT_dil, VT_conc
    obs_recorder = np.array(obs_recorder).astype(np.float32) # SR, EC, Cdil_f, Cconc_f
    # add to dataframes
    both = np.concatenate((action_recorder, obs_recorder), axis=1)
    columns = ['N', 'Estack', 'T_tot', 'VT_dil', 'VT_conc', 'SR', 'EC', 'Cdil_f', 'Cconc_f']
    df = pd.DataFrame(both, columns=columns)
    # write the dataframe to a csv file
    df.to_csv(write_path + f'/test_{id}_reward_{reward_id}.csv', index=False, header=True)

    return df
