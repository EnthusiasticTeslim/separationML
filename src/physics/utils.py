import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import argparse
import os

MW_Cl = 35.45 # Molecular weight of Cl, g/mol
MW_Na = 23 # Molecular weight of Na, g/mol

# NaCl to Cl and Cl to NaCl
def NaCl2Cl(c):
    """ Convert NaCl concentration (in ppm) to Cl concentration (in ppm) """
    
    return c * MW_Cl / (MW_Cl + MW_Na)


def ppm2molm3(ppm, mw):
    """ convert ppm to mol/m3 """

    return (ppm / mw / 1000) * 1000

def simulate(model, dt: int = 10, method: str = 'solve_ivp'): 
    """ Simulate the electrodialysis process
    Parameters
    ----------
        model: instance of the electrodialysis class
        dt: number of time steps
    Returns
    -------
    result: dictionary containing the results of the simulation 
    """

    # Initial conditions
    # Cdil0 = ppm2molm3(ode.feed)
    # Cconc0 = ppm2molm3(ode.feed)
    Feed_Cl_only = NaCl2Cl(model.feed) # Feed of Cl only, ppm
    Cconc0 = ppm2molm3(ppm=Feed_Cl_only, mw=MW_Cl) # Initial concentration of concentrate
    Cdil0 = ppm2molm3(ppm=Feed_Cl_only, mw=MW_Cl) # Initial concentration of diluate
    # Initial conditions
    C0 = np.array([Cconc0, Cdil0, Cconc0, Cdil0] )
    # solve ODE
    teval = np.linspace(0, model.T_tot, dt)
    if method == 'odeint':
        C = sp.integrate.odeint(func=model.dCdt, y0=C0, t=teval, tfirst=True) # Solve the ODEs
        Cconc = np.array(C[:, 0])
        Cdil = np.array(C[:, 1])
        Cconc_in = np.array(C[:, 2])
        Cdil_in = np.array(C[:, 3])
    elif method == 'solve_ivp':
        sol = sp.integrate.solve_ivp(fun=model.dCdt, t_span=(0, model.T_tot), y0=C0, t_eval=teval, method='LSODA')
        Cconc = np.array(sol.y[0])
        Cdil = np.array(sol.y[1])
        Cconc_in = np.array(sol.y[2])
        Cdil_in = np.array(sol.y[3])
    else:
        raise ValueError('Invalid method: choose either odeint or solve_ivp')
    
    j_values = [] # Append the calculated current density 'j' to the array for each time step
    for i in range(len(teval)):
        j = model.compute_j(Cconc[i], Cdil[i]) # A/m2
        j_values.append(j)
    
    j_values = np.array(j_values) # A/m2

    cases  = {  
                'SR': (1 - Cdil[-1] / Cdil0), # Separation efficiency, unitless
                'EC': 0, # Energy consumption, kJ/kg
                'Cconc': Cconc, # Concentration in effluent concentrate stream, mol/m3
                'Cdil': Cdil, # Concentration in effluent diluate stream, mol/m3
                'Cconc_in': Cconc_in,  # Concentration in inlet concentrate stream, mol/m3
                'Cdil_in': Cdil_in, # Concentration in inlet diluate stream, mol/m3
                'j_values': j_values / 10, # Current density, mA/cm2
                'times': teval/3600 # Time in hours
                }

    return cases 

def objective(
        estimator: callable,
        params: dict,
        action: list,
        steps: int,
        is_scipy: bool=False,
        method: str='solve_ivp'):
    '''Objective function to maximize
    Parameters:
    ----------
    estimator: object, instance of the electrodialysis class
    params: dict, dictionary containing the parameters of the simulation
    action: list, [N, Estack, T_tot]
    steps: int, number of timesteps
    is_scipy: bool, True if using scipy optimization

    Returns:
    -------
    float: if not scipy, separation efficiency and final concentration in diluate stream else -SR'''

    N, Estack, T_tot = action
    N = int(round(N)) # round to the closest integer value #int(N) + 1  
    # copy self.params
    params = params.copy()
    # update the parameters
    params['N'] = N
    params['Estack'] = Estack
    params['T_tot'] = T_tot
    # add the parameters to model
    estimator.set_params(params)
    # simulate the electrodialysis process & compute separation efficiency as reward
    result  = simulate(model=estimator, dt=steps, method=method)
    # get the separation efficiency and final concentration
    SR = result['SR'] # max is 99.9%
    final_concentration = result['Cdil'][-1] # final concentration in diluate stream
    
    if is_scipy:
        return -SR
    else:
        return SR, final_concentration

def scipy_optimization(
        estimator: callable, 
        params: dict, 
        steps: int, 
        bounds: list,
        initial_guess: list):
    '''Perform scipy optimization to find the optimal parameters
    Parameters:
    ----------
    estimator: object, instance of the electrodialysis class
    params: dict, dictionary containing the parameters of the simulation
    steps: int, number of timesteps
    bounds: list, bounds for the parameters
    initial_guess: list, initial guess for the parameters

    Returns:
    -------
    None'''

    # Perform the optimization
    result = minimize(objective, x0=initial_guess, bounds=bounds, args=(estimator, params, steps, True))

    # Extract the optimal parameters
    sp_N, sp_V, sp_T_tot = result.x

    print(f"N = {sp_N:.4f}, V = {sp_V:.4f}, T_tot = {sp_T_tot:.4f}")
    print(f"reward: {-result.fun:.4f}")


# grid search
def grid_search(
        estimator: callable,
        params: dict,
        args: argparse.Namespace,
        lower_bounds: list =[1, 0.1, 0],
        upper_bounds: list =[80, 2.0, 180],
        steps: list = [1, 0.1, 5]
        ):
    '''Perform grid search to find the best parameters
    Parameters:
    ----------
    estimator: object, instance of the electrodialysis class
    params: dict, dictionary containing the parameters of the simulation
    args: object, instance of the argparse class
    
    Returns:
    -------
    None'''

    print("\nPerforming grid search...")
    best_grid_value = 0
    best_grid_params = None

    for N in np.arange(lower_bounds[0], upper_bounds[0]+1, steps[0]): # 1 to 80
        for V in np.arange(lower_bounds[1], upper_bounds[1]+0.1, steps[1]): # 0.1 to 2 in steps of 0.1
            for T_tot in np.arange(lower_bounds[2], upper_bounds[2]+5, steps[2]): # 0 to 180 in steps of 5
                params['N'] = N
                params['Estack'] = V
                params['T_tot'] = T_tot
                estimator.set_params(params=params)
                reward = simulate(model=estimator, dt=args.steps)['SR']
                if reward > best_grid_value:
                    best_grid_value = reward
                    best_grid_params = (N, V, T_tot)

    print('Best Search Parameters:')
    print(f"N={best_grid_params[0]}, V={best_grid_params[1]:.4f}, T_tot={best_grid_params[2]:.4f}")
    print(f"reward: {best_grid_value:.6f}")


def save_model(model, cwd: str, args: argparse.Namespace):
    '''Save the reinforcement learning model
    Parameters:
    ----------
    model: object, instance of the model
    cwd: str, current working directory
    args: object, instance of the argparse class

    Returns:
    -------
    None
    '''
    if "src" in cwd:
        print('Model saved in parent folder')
        saved_path = f"/{cwd.strip('/src')}/models/{args.RL}_electrodialysis"
    else:        
        print('Model saved in current folder')
        saved_path = f"/{cwd}/models/{args.RL}_electrodialysis"
    model.save(saved_path)
    print(f"Model saved in {saved_path}")


def load_model(cwd: str, args: argparse.Namespace):  

    if 'src' in cwd and os.path.exists(f"/{cwd.strip('/src')}/models/{args.RL}_electrodialysis.zip"):
        path  = f"/{cwd.strip('/src')}/models/{args.RL}_electrodialysis"
    elif os.path.exists(f"{cwd}/models/{args.RL}_electrodialysis.zip"):
        path = f"/{cwd}/models/{args.RL}_electrodialysis"
    else:
        raise FileNotFoundError('Model not found')
    print(f"Model loaded from {path}")
    
    return path


def plot_data(ode, result, exp=None):

    """ Plot the results of the simulation
    Parameters
    ----------
    ode : object
        instance of the electrodialysis class
    result : dict
        dictionary containing the results of the simulation
    exp : dict
        dictionary containing the experimental data
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # outlet streams
    axs[0].plot(result['times'], result['Cconc']*ode.MW_Cl, color = 'r')
    axs[0].plot(result['times'], result['Cdil']*ode.MW_Cl, color = 'b')
    if exp is not None:
        axs[0].plot(exp['time (hr)'], exp['Cconc (ppm)'], 'ro')
        axs[0].plot(exp['time (hr)'], exp['Cdil (ppm)'], 'b*')
    axs[0].set_xlabel(r'$\rm Time\ (h)$', labelpad=5, fontsize='x-large')
    axs[0].set_ylabel(r'$\rm Outlet\ Concentration\ (mol/m^3)$', labelpad=5, fontsize='x-large')
    # inlet streams
    if exp is not None:
        axs[1].plot(exp['time (hr)'], exp['Cconc_in (ppm)'], 'ro')
        axs[1].plot(exp['time (hr)'], exp['Cdil_in (ppm)'], 'b*')

    axs[1].plot(result['times'], result['Cconc_in']*ode.MW_Cl, color = 'r')
    axs[1].plot(result['times'], result['Cdil_in']*ode.MW_Cl, color = 'b')
    axs[1].set_xlabel('Time (h)', labelpad=5, fontsize='x-large')
    axs[1].set_ylabel(r'$\rm Inlet\ Concentration\ (mol/m^3)$', labelpad=5, fontsize='x-large')
    # current density
    if exp is not None:
        axs[2].plot(exp['time (hr)'], exp['j(mA/cm2)'], 'ko')
    axs[2].plot(result['times'], result['j_values'], color = 'k')
    axs[2].set_ylabel(r'$\rm Current\ density\ (mA/cm^2)$', labelpad=5, fontsize='x-large')
    axs[2].set_xlabel(r'$\rm Time\ (h)$', labelpad=5, fontsize='x-large')

    plt.tight_layout()
    plt.show()