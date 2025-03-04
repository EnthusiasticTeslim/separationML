# Author: Teslim Olayiwola
# Date: 2024-11-10
# Description: Custom environment for the electrodialysis simulation

import numpy as np
from tqdm import tqdm
from .utils import NaCl2Cl, ppm2molm3, act, rho

class NumericalModel:
    '''electrodialysis model based on the electrodialysis equation'''
    def __init__(self, print_info=False):
        self.print_info = print_info # print the parameters if True
        self.F = 96485 # Faraday's constant, C/mol
        self.R = 8.314 # Gas constant, J/mol-K
        self.z = 1 # No of electrons transferred
        self.t_minus, self.t_plus = 0.60, 0.40 # unitless
        self.MW_Cl = 35.45 # Molecular weight of Cl, g/mol
        self.MW_Na = 23 # Molecular weight of Na, g/mol
        self.Ean_minus_Ecat = 1.5 # Anode and cathode voltage drop, assumed constant
        self.tolerance = 3 # tolerance for the current density

    def set_params(self, params, constant_voltage=True): #
        # N, Vconc, Vdil, Qconc, Qdil, phi, Da, Dc, km, La, Lc, Ldil, Lconc, A, ra, rc, Etot, Feed, T_tot, T
        self.feed = params['Feed'] # Feed concentration (NaCL) in ppm
        self.Vk_dil = params['Ldil'] * params['A'] # computed diluate tank volume, m^3
        self.Vk_conc = params['Lconc'] * params['A'] # computed concentrate tank volume, m^3
        self.N = params['N'] # No of cell pairs, unitless
        self.ra = params['ra'] # AEM electrical resistance Ohm-m^2,
        self.rc = params['rc'] # CEM electrical resistance Ohm-m^2
        self.Estack = params['Estack'] # Volts in a stack
        self.Qconc = params['Qconc']/(60*60) # Flow rate dil. chamber m^3/hr to m^3/s
        self.Qdil = params['Qdil']/(60*60) # Flow rate conc. chamber m^3/hr to m^3/s
        self.phi = params['phi'] # Current efficiency
        self.Da = params['Da'] # Avg diffusion coeff of NaCl in AEM (Ortiz. et. al) m^2/s
        self.Dc = params['Dc'] # Avg diffusion coeff of NaCl in CEM (Ortiz. et. al) m^2/s
        self.km = params['km'] # Mass transfer coeff, m/s
        self.La = params['La'] # AEM thickness, m
        self.Lc = params['Lc'] # CEM thickness, m
        self.Ldil = params['Ldil'] # Diluate channel thickness, m
        self.Lconc = params['Lconc'] # Concentrate channel thickness, m
        self.A = params['A'] # Membrane area, m^2
        self.VT_conc = params['VT_conc'] # concentrate tank volume, m^3
        self.VT_dil = params['VT_dil'] # diluate tank volume, m^3
        self.T_tot = params['T_tot']*60 # Total time, hr to min
        self.permAEM = params['permAEM'] # AEM permselectivity
        self.permCEM = params['permCEM'] # CEM permselectivity
        self.constant_voltage = constant_voltage # constant current or not
        if self.constant_voltage is False:
            self.current = params['current']

        ## Compute the initial concentration of the concentrate and diluate
        # Feed of Cl from NaCl, ppm
        Feed_Cl_only = NaCl2Cl(self.feed) 
        # Initial concentration of concentrate and diluate, #(Feed_Cl_only / self.MW_Cl / 1000) * 1000
        self.Cconc0 = ppm2molm3(Feed_Cl_only, self.MW_Cl)  
        self.Cdil0 = ppm2molm3(Feed_Cl_only, self.MW_Cl) 

        if self.print_info:
            tqdm.write('Parameters set successfully')

    def __repr__(self):
        if self.print_info:
            return 'Parameters are:\n Feed: {} mol/m3\n Vk_dil: {} m^3\n Vk_conc: {} m^3\n N: {}\n ra: {} Ohm-m^2\n rc: {} Ohm-m^2\n V: {} V\n Qdil: {} m^3/s\n Qconc: {} m^3/s\n phi: {}\n Da: {} m^2/s\n Dc: {} m^2/s\n km: {}\n La: {} m\n Lc: {} m\n Ldil: {} m\n Lconc: {} m\n A: {} m^2\n VT_conc: {} m^3\n VT_dil: {} m^3\n Time: {} min\n pAEM: {}\n pCEM: {}'.format(self.feed, self.Vk_dil, self.Vk_conc, self.N, self.ra, self.rc, self.Estack, self.Qdil, self.Qconc, self.phi, self.Da, self.Dc, self.km, self.La, self.Lc, self.Ldil, self.Lconc, self.A, self.VT_conc, self.VT_dil, self.T_tot, self.permAEM, self.permCEM)
        else:
            return 'Parameters are hidden'

    def compute_current_density(self, Cconc, Cdil):

        T = 273.15 + 25 # K
        j = 0.001 # A/m2 initial guess: Variable to store the calculated value of j
        while True:

            Cwadil = Cdil - self.phi * j / (self.z * self.F * self.km) * (1 - self.t_minus)
            Cwaconc = Cconc + self.phi * j / (self.z * self.F * self.km) * (1 - self.t_minus)
            Cwcdil = Cdil - self.phi * j / (self.z * self.F * self.km) * (1 - self.t_plus)
            Cwcconc = Cconc + self.phi * j / (self.z * self.F * self.km) * (1 - self.t_plus)

            a_conc_wc = act(Cwcconc)
            a_dil_wc = act(Cwcdil)
            a_conc_wa = act(Cwaconc)
            a_dil_wa = act(Cwadil)

            Emem = (self.R * T / self.F) * (self.permAEM*np.log((a_conc_wa * Cwaconc)
                                                                      / (a_dil_wa * Cwadil))
                                                + self.permCEM*np.log((a_conc_wc * Cwcconc)
                                                                      / (a_dil_wc * Cwcdil)))

            R_dil = rho(Cdil, self.Ldil) # Resistance of diluate, ohm-m2
            R_conc = rho(Cconc, self.Lconc) # Resistance of concentrate, ohm-m2
            # compute current density in A/m2
            calculated_j = (self.N*self.Estack - self.Ean_minus_Ecat - self.N * Emem) / (self.N * (R_dil + R_conc + self.ra + self.rc))
            # Check if the calculated j is close enough to the previous value
            if abs(calculated_j - j) < self.tolerance:
                j = calculated_j
                break

            j = calculated_j

        return j

    def dCdt(self, t, C):
        '''
            params:
                Cconc: Concentrate concentration at outlet
                Cdil: Concentrate concentration at outlet
                Cconc_in: Concentrate concentration at inlet
                Cdil_in: Diluate concentration at inlet
            return:
                DconcDt: 1st diff of Concentrate concentration wrt time
                DdilDt: 1st diff of Diluate concentration wrt time
                Dconc_inDt: 1st diff of Conc concentratn wrt time @ inlet
                Ddil_inDt: 1st diff of Diluate concentration wrt time @ inlet

        '''
    
        Cconc, Cdil, Cconc_in, Cdil_in = C[0], C[1], C[2], C[3]
        # compute j
        if self.constant_voltage:
            j = self.compute_current_density(Cconc, Cdil)
        else:
            j = self.current
        ## 1.compute concentration at the membrane interface
        # Concentration at the anionic membrane interface with the diluate
        Cwadil = Cdil - self.phi * j * (1 - self.t_minus) / (self.z * self.F * self.km)
        # Concentration at the anionic membrane interface with the concentrate
        Cwaconc = Cconc + self.phi * j * (1 - self.t_minus)/ (self.z * self.F * self.km)
        # Concentration at the cationic membrane interface with the diluate
        Cwcdil = Cdil - self.phi * j * (1 - self.t_plus) / (self.z * self.F * self.km)
        # Concentration at the cationic membrane interface with the concentrate
        Cwcconc = Cconc + self.phi * j * (1 - self.t_plus) / (self.z * self.F * self.km)
        ## 2. compute the concentration at the outlet, defined by equation 1a and 1b
        # ODE of mass balance in concentrate tank
        DconcDt = (self.Qconc * (Cconc_in - Cconc)
                        + self.N * self.phi * j * self.A / (self.z * self.F)
                        - self.N * self.A * self.Da * (Cwaconc - Cwadil) / self.La
                        - self.N * self.A * self.Dc * (Cwcconc - Cwcdil) / self.Lc) / (self.N * self.Vk_conc)
        # ODE of mass balance in diluate tank
        DdilDt = (self.Qdil * (Cdil_in - Cdil)
                        - self.N * self.phi * j * self.A / (self.z * self.F)
                        + self.N * self.A * self.Da * (Cwaconc - Cwadil) / self.La
                        + self.N * self.A * self.Dc * (Cwcconc - Cwcdil) / self.Lc) / (self.N * self.Vk_dil)

        # ODE of mass balance in concentrate recirculation tank
        Dconc_inDt = self.Qconc * (Cconc - Cconc_in)/self.VT_conc
        # ODE of mass balance in diluate recirculation tank
        Ddil_inDt = self.Qdil * (Cdil - Cdil_in)/self.VT_dil

        # 3. return the derivatives
        return [DconcDt, DdilDt, Dconc_inDt, Ddil_inDt]