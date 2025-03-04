## Reinforcement learning 
In this work, the reinforcement learning (RL) framework considered the following concepts:
- Agent: The learner or decision-maker.
- Environment: Everything the agent interacts with.
- State: A specific situation in which the agent finds itself.
- Action: All possible moves the agent can make.
- Reward: Feedback from the environment based on the action taken.


### Environment 
A simulation model of the electrodialysis system that can predict the next state given the current state and action. This is a physics-based model such that action (i.e. change in N, V, T_tot) results in effluent concentration and current (or voltage) which is then used to compute the reward. 


### Reward 
Here, we adopted two reward functions:

    a) separation efficiency, only; maximized. 
    b) separation efficiency and energy consumption with goal of maximizing the separation and minimizing the energy. 


### Action Space 
Here, there are many parameters considered as what the agent tunes

    a) N: Number of cell pairs (discrete)
    Range: Typically 15 to 80

    b) V: Applied voltage 

    Range: 0.1 to 2.0 volts

    c) T_tot: Total operation time 

    Range: 45 to 120 min.

    d) VT_dil: Diluate reservoir tank volume
    Range: 0.1  - 2.0 m^3

    e) VT_conc: Concentrate reservoir tank volume
    Range: 0.01  - 0.5 m^3


### State

Here, there are many parameters considered as what the agent mov

    a) SR: salt removal efficiency
    Range: Typically 0 to 1

    b) EC: Energy consumed 

    Range: >0 KWh/m3 (set max 10)

    c) Cdil_f: final dulate concentration
    Range: > 0 mol/m3  (set max 1000)

    d) Cconc_f: final Concentrate concentration
    Range: > 0 mol/m3 (set max 1000)


### Algorithm
Choose an algorithm (from `from stable_baseline3 import SAC, PPO, A2C`) that can handle mixed discrete-continuous action spaces. Options include:
- Soft Actor-Critic (SAC)
- Proximal Policy Optimization (PPO)
-  Advantage Actor Critic (A2C).


### Training Process:
The RL agent would interact with the environment (simulated electrodialysis system) by:
- Observing the current state
- Choosing actions (N, V, T_tot, VT_dil, VT_conc)
- Receiving a reward based on the system's performance
- Updating its policy to maximize long-term rewards


### Strategies
The implementation steps are as follows:
- [x] Develop a accurate simulation model of the electrodialysis system
- [x] Define the RL components (state space, action space, reward function)
- [x] Implement the chosen RL algorithm on single and multi-objective scenarios.
- [x] Train the agent on the simulated environment
- [x] Carefully test on the real system, starting with a limited operational range