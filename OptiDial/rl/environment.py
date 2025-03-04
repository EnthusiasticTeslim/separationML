# Author: Teslim Olayiwola
# Date: 2024-12-10
# Description: Custom environment for the electrodialysis simulation
import numpy as np
import gymnasium as gym

class ENV(gym.Env):
    def __init__(self, 
                 rl_metrics, 
                 reward_function,
                 reward_weight = None,
                 max_steps: int = 1000):
        """ Initialize the custom environment
            Parameters
            ----------
                estimator : object, instance of the electrodialysis class
                params : dict, dictionary containing the parameters of the simulation
                steps: int, number of timesteps
                max_conc: float, maximum possible concentration in diluate stream
            Returns
            -------
                object: instance of the custom environment
            Credits
            -------
            https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb
            """
        self.rl_metrics = rl_metrics
        self.reward_function = reward_function
        self.reward_weight = reward_weight
        self.max_steps = max_steps
        self.state = None
        self.steps = 0

        # define the action space: 
        # N, Estack, T_tot, VT_dil, VT_conc
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1, -1, -1]), 
                                           high=np.array([1, 1, 1, 1, 1]), 
                                           dtype=np.float32)

        # define the observation space: 
        # SR (0 - 1), EC (0 and above), Cdil_f (0 and above), Cconc_f (0 and above)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]),
                                                high=np.array([1, 10, 1000, 1000]),
                                                dtype=np.float32)


    def step(self, action):
        """ Execute one time step within the environment. 
        Parameters
        ----------
            action: list, [N, Estack, T_tot, VT_dil, VT_conc]
        Returns
        -------
            observation: list, [SR, EC, Cdil_f, Cconc_f]
            reward: float, 
            done: bool, True if the episode is done
            truncated: bool, False
            info: dict, additional information
            
        """
        # increment the number of steps
        self.steps += 1
        # clip the action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # compute the desalinization metrics
        SR, EC, Cdil_f, Cconc_f, invalid = self.rl_metrics.operate(action=action)
        # set state
        self.state = np.array([SR, EC, Cdil_f, Cconc_f]).astype(np.float32)
        # get the reward  and check if the episode is done
        # print(invalid, SR, EC)
        if not invalid:
            reward = self.reward_function(SR=SR, EC=EC, weight=self.reward_weight)
            truncated = self.steps >= self.max_steps # # we limit the number of steps
            terminated = bool(SR >= 0.99 or EC <= 0) # # we terminate if SR is greater than 0.99 or EC is less than 0
        else: # assign a big negative reward if the simulation is invalid
            reward = -1e3
            terminated = True
            truncated = True
        
        info = {
            'outlet_dil_conc': Cdil_f,
            'outlet_conct_conc': Cconc_f,
            'SR': SR,
            'EC': EC,
            'steps': self.steps
        }

        return self.state, reward, terminated, truncated, info 

    def reset(self, seed=None):
        """
        Reset the environment to an initial state
        Returns
        -------
            state : np.array, Initial observation.
            info : dict, Additional diagnostic information.
        """
        super().reset(seed=seed)

        # sample from the action space
        init_action = self.action_space.sample()
        # # compute the desalinization metrics SR, EC, Cdil_f, Cconc_f, invalid
        try:
            metrics  = self.rl_metrics.operate(action=init_action)[:-1]
        except:
            metrics = [0, 10, 1000, 1000]
        # Initialize state self.observation_space.sample() #
        self.state = np.array(metrics).astype(np.float32)
        # reset the number of steps
        self.steps = 0
        # info
        info = {
            'outlet_diluate_concentration': metrics[2],
            'outlet_concentrate_concentration': metrics[3],
            'SR': metrics[0],
            'EC': metrics[1],
            'steps': self.steps
        }

        return self.state, info

    def render(self):
        """
        Render the environment
        """
        pass