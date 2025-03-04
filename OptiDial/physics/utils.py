import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson, solve_ivp, odeint
import argparse
import os

MW_Cl = 35.45 # Molecular weight of Cl, g/mol
MW_Na = 23 # Molecular weight of Na, g/mol

# NaCl to Cl and Cl to NaCl
def NaCl2Cl(c):
    """ Convert NaCl concentration (in ppm) to Cl concentration (in ppm) 
    Parameters
    ----------
        c: NaCl concentration in ppm
    Returns
    -------
        c: Cl concentration in ppm
    """
    
    return c * MW_Cl / (MW_Cl + MW_Na)


def ppm2molm3(ppm, mw):
    """ convert ppm to mol/m3 
    Parameters
    ----------
        ppm: concentration in ppm
        mw: molecular weight, g/mol
    Returns
    -------
        molm3: concentration in mol/m3
    """

    return (ppm / mw / 1000) * 1000

def molm32ppm(molm3, mw):
    """ convert mol/m3 to ppm 
    Parameters
    ----------
        molm3: concentration in mol/m3
        mw: molecular weight, g/mol
    Returns
    -------
        ppm: concentration in ppm
    """
    
    return (molm3 * mw / 1000) * 1000

def act(c):
    '''
    Compute the activity of the ion in the solution
    Parameters:
    ----------
        c: concentration of ion, c in mol/m3
    Returns:
    -------
        act: activity of the ion, unitless
    extra
    -------
        remove the 1e-3 if C is in mol/L
        ref: https://www.sciencedirect.com/science/article/abs/pii/S0011916417325262
    '''
    negLogAct = (0.5065 * np.sqrt(c * 1e-3)) / (1 + 1.298 * np.sqrt(c * 1e-3)) - (0.039 * c * 1e-3)
    act = 10 ** (-negLogAct)
    return act

def rho(C, L):
    ''' Compute the resistance of the solution
    Parameters:
    ----------
        L: length of the channel, m
        C: concentration of the solution, mol/m3.
        u: temperature-independent parameter, Angstrom
        k: molar conductivity of the diluated solution, S/m

    Returns:
    -------
        rh: resistance of the solution, ohm-m2 '''

    B0, B1, B2, lambda0, u = [0.3286, 0.2289, 60.32, 126.45, 4.0]

    lambda_ = lambda0 - (B1 * lambda0 + B2) * np.sqrt(C * 1e-3) / (1 + B0 * u * np.sqrt(C * 1e-3)) # cm^2-S/mol, c (mol/m3)
    k = (C * lambda_ * 1e-4) # S-m^2/mol, added 1e-4 to convert cm^2-S/mol to m^2-S/mol
    rh = L / k # ohm-m2
    return  rh

def SR(Cdil_f, Cdil_i):
    """ Separation efficiency 
    Parameters
    ----------
        Cdil_f: final concentration in diluate stream, mol/m3
        Cdil_i: initial concentration in diluate stream, mol/m3
    Returns
    -------
        SR: separation efficiency, unitless
    """
    
    return 1 - (Cdil_f / Cdil_i)

def EC(N, Ecell, j, A, teval, VTdil):
    """ Energy consumption in KWh/kg 
    Parameters
    ----------
        N: number of cell pairs
        Ecell: cell potential, V
        j: current density, A/m2
        A: membrane area, m2
        teval: time array
        VTdil: volume of diluate stream, m3
    Returns
    -------
        EC_per_whm3: energy consumption in KWh/m3
        """
    
    EC_J = N * Ecell * A * simpson(j, teval) # A*V = J
    EC_per_whm3 = 2.78 * 1e-4 * EC_J / (VTdil) # J/m3 to Wh/m3

    return EC_per_whm3/1000 # Wh/m3 to KWh/m3

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
    Feed_Cl_only = NaCl2Cl(model.feed) # Feed of Cl only, ppm
    Cconc0 = ppm2molm3(ppm=Feed_Cl_only, mw=MW_Cl) # Initial concentration of concentrate
    Cdil0 = ppm2molm3(ppm=Feed_Cl_only, mw=MW_Cl) # Initial concentration of diluate
    # Initial conditions
    C0 = np.array([Cconc0, Cdil0, Cconc0, Cdil0] )
    # solve ODE
    teval = np.linspace(0, model.T_tot, dt)
    if method == 'odeint':
        C = odeint(func=model.dCdt, y0=C0, t=teval, tfirst=True) # Solve the ODEs
        Cconc = np.array(C[:, 0])
        Cdil = np.array(C[:, 1])
        Cconc_in = np.array(C[:, 2])
        Cdil_in = np.array(C[:, 3])
    elif method == 'solve_ivp':
        sol = solve_ivp(fun=model.dCdt, t_span=(0, model.T_tot), y0=C0, t_eval=teval, method='LSODA')
        Cconc = np.array(sol.y[0])
        Cdil = np.array(sol.y[1])
        Cconc_in = np.array(sol.y[2])
        Cdil_in = np.array(sol.y[3])
    else:
        raise ValueError('Invalid method: choose either odeint or solve_ivp')
    
    j_values = [] # Append the calculated current density 'j' to the array for each time step
    for i in range(len(teval)):
        j = model.compute_current_density(Cconc[i], Cdil[i]) # A/m2
        j_values.append(j)
    
    j_values = np.array(j_values) # A/m2

    cases  = {  
                'SR': SR(Cdil_f=Cdil[-1], Cdil_i=Cdil0), # Separation efficiency, unitless
                'WR': (Cconc[0] - Cconc[-1]) / (Cdil[-1] - Cdil[0]), # Water recovery, unitless
                'EC': EC(N=model.N, Ecell=model.Estack, j=j_values, A=model.A, teval=teval, VTdil=model.VT_dil), # Energy consumption, KWh/m3
                'Cdil_f': Cdil[-1], # Final concentration in diluate stream, mol/m3
                'Cconc_f': Cconc[-1], # Final concentration in concentrate stream, mol/m3
                'Cconc_i': Cconc[0], # Initial concentration in concentrate stream, mol/m3
                'Cdil_i': Cdil[0], # Initial concentration in diluate stream, mol/m3
                'Cconc': Cconc, # Concentration in effluent concentrate stream, mol/m3
                'Cdil': Cdil, # Concentration in effluent diluate stream, mol/m3
                'Cconc_in': Cconc_in,  # Concentration in inlet concentrate stream, mol/m3
                'Cdil_in': Cdil_in, # Concentration in inlet diluate stream, mol/m3
                'j_values': j_values / 10, # Current density, mA/cm2
                'times': teval/3600 # Time in hours
                }

    return cases 

def sensitivity_analysis(
                        model, # electrochemical model
                         parameters, # parameters of the simulation
                         var: str, # parameter to vary
                         var_list: list, # list of values to vary
                         time_steps: int = 20 # number of time steps
                         ):
    
    ''' Perform sensitivity analysis
    Parameters:
    ----------
        model: object, instance of the electrodialysis class
        parameters: dict, dictionary containing the parameters of the simulation
        var: str, parameter to vary
        var_list: list, list of values to vary
        time_steps: int, number of time steps

    Returns:
    -------
        SR_list: list, list of separation efficiency values
        EC_list: list, list of energy consumption values
    '''
    
    # create a copy of the parameters
    parameters = parameters.copy()
    SR_list = []
    EC_list = []
    
    for param in var_list:
        parameters[f'{var}'] = param
        # update the parameter in the model
        model.set_params(parameters)
        result = simulate(model=model, dt=time_steps)
        SR_list.append(result['SR'])
        EC_list.append(result['EC'])

    SR_list = np.array(SR_list)
    EC_list = np.array(EC_list)

    return SR_list, EC_list

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