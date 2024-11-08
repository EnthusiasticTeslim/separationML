import numpy as np
from tqdm import tqdm
import torch
from .utils import NaCl2Cl, ppm2molm3

class electrodialysis:
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

    def set_params(self, params, incorporate_init=False, torch=False, constant_voltage=True): #
        # N, Vconc, Vdil, Qconc, Qdil, phi, Da, Dc, km, La, Lc, Ldil, Lconc, A, ra, rc, Etot, Feed, T_tot, T
        self.torch = torch # use torch tensors if True
        self.feed = params['Feed'] # Feed concentration (NaCL) in ppm
        self.Vk_dil = params['Ldil'] * params['A'] # computed diluate tank volume, m^3
        self.Vk_conc = params['Lconc'] * params['A'] # computed concentrate tank volume, m^3
        self.N = params['N'] # No of cell pairs
        self.ra = params['ra'] # AEM electrical resistance Ohm-m^2,
        self.rc = params['rc'] # CEM electrical resistance Ohm-m^2
        self.Estack = params['Estack'] # Volts in a stack
        self.Qconc = params['Qconc']/(60*60) # Flow rate dil. chamber m^3/hr
        self.Qdil = params['Qdil']/(60*60) # Flow rate conc. chamber m^3/hr
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
        self.T_tot = params['T_tot']*60 # Total time, min
        self.permAEM = params['permAEM'] # AEM permselectivity
        self.permCEM = params['permCEM'] # CEM permselectivity
        self.constant_voltage = constant_voltage # constant current or not
        if self.constant_voltage is False:
            self.current = params['current']


        if incorporate_init is True:
            Feed_Cl_only = NaCl2Cl(self.feed) # Feed of Cl only, ppm
            self.Cconc0 = ppm2molm3(Feed_Cl_only, self.MW_Cl) #(Feed_Cl_only / self.MW_Cl / 1000) * 1000 # Initial concentration of concentrate
            self.Cdil0 = ppm2molm3(Feed_Cl_only, self.MW_Cl) #(Feed_Cl_only / self.MW_Cl / 1000) * 1000 # Initial concentration of diluate

        if self.print_info:
            tqdm.write('Parameters set successfully')

    def __repr__(self):
        if self.print_info:
            return 'Parameters are:\n Feed: {} mol/m3\n Vk_dil: {} m^3\n Vk_conc: {} m^3\n N: {}\n ra: {} Ohm-m^2\n rc: {} Ohm-m^2\n V: {} V\n Qdil: {} m^3/s\n Qconc: {} m^3/s\n phi: {}\n Da: {} m^2/s\n Dc: {} m^2/s\n km: {}\n La: {} m\n Lc: {} m\n Ldil: {} m\n Lconc: {} m\n A: {} m^2\n VT_conc: {} m^3\n VT_dil: {} m^3\n Time: {} min\n pAEM: {}\n pCEM: {}'.format(self.feed, self.Vk_dil, self.Vk_conc, self.N, self.ra, self.rc, self.Estack, self.Qdil, self.Qconc, self.phi, self.Da, self.Dc, self.km, self.La, self.Lc, self.Ldil, self.Lconc, self.A, self.VT_conc, self.VT_dil, self.T_tot, self.permAEM, self.permCEM)
        else:
            return 'Parameters are hidden'

    def act(self, c):
        '''
        params:
            concentration of ion, c in mol/m3
        return:
            activity of the ion, unitless
        extra:
            # remove the 1e-3 if C is in mol/L
            # ref: https://www.sciencedirect.com/science/article/abs/pii/S0011916417325262
        '''
        negLogAct = (0.5065 * np.sqrt(c * 1e-3)) / (1 + 1.298 * np.sqrt(c * 1e-3)) - (0.039 * c * 1e-3)
        act = 10 ** (-negLogAct)
        return act

    def rho(self, C, L):
        '''
        params:
            L: length of the channel, m
            A: cross section of the channel, m^2
            C: concentration of the solution, mol/m3.
            u: temperature-independent parameter, Angstrom
            k: molar conductivity of the diluated solution, S/m
        return:
            rh: resistance of the solution, ohm-m2 '''

        B0, B1, B2, lambda0, u = [0.3286, 0.2289, 60.32, 126.45, 4.0]

        lambda_ = lambda0 - (B1 * lambda0 + B2) * np.sqrt(C * 1e-3) / (1 + B0 * u * np.sqrt(C * 1e-3)) # cm^2-S/mol, c (mol/m3)
        k = (C * lambda_ * 1e-4) # S-m^2/mol, added 1e-4 to convert cm^2-S/mol to m^2-S/mol
        rh = L / k # ohm-m2
        return  rh

    def compute_j(self, Cconc, Cdil):

        T = 273.15 + 25 # K
        j = 0.001 # A/m2 initial guess: Variable to store the calculated value of j
        while True:

            Cwadil = Cdil - self.phi * j / (self.z * self.F * self.km) * (1 - self.t_minus)
            Cwaconc = Cconc + self.phi * j / (self.z * self.F * self.km) * (1 - self.t_minus)
            Cwcdil = Cdil - self.phi * j / (self.z * self.F * self.km) * (1 - self.t_plus)
            Cwcconc = Cconc + self.phi * j / (self.z * self.F * self.km) * (1 - self.t_plus)

            a_conc_wc = self.act(Cwcconc)
            a_dil_wc = self.act(Cwcdil)
            a_conc_wa = self.act(Cwaconc)
            a_dil_wa = self.act(Cwadil)

            Emem = (self.R * T / self.F) * (self.permAEM*np.log((a_conc_wa * Cwaconc)
                                                                      / (a_dil_wa * Cwadil))
                                                + self.permCEM*np.log((a_conc_wc * Cwcconc)
                                                                      / (a_dil_wc * Cwcdil)))

            R_dil = self.rho(Cdil, self.Ldil) # Resistance of diluate, ohm-m2
            R_conc = self.rho(Cconc, self.Lconc) # Resistance of concentrate, ohm-m2
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
            j = self.compute_j(Cconc, Cdil)
        else:
            j = self.current
        # compute concentration at the membrane interface
        Cwadil = Cdil - self.phi * j * (1 - self.t_minus) / (self.z * self.F * self.km)
        Cwaconc = Cconc + self.phi * j * (1 - self.t_minus)/ (self.z * self.F * self.km)
        Cwcdil = Cdil - self.phi * j * (1 - self.t_plus) / (self.z * self.F * self.km)
        Cwcconc = Cconc + self.phi * j * (1 - self.t_plus) / (self.z * self.F * self.km)
        # compute the concentration at the outlet, defined by equation 1a and 1b
        DconcDt = (self.Qconc * (Cconc_in - Cconc)
                        + self.N * self.phi * j * self.A / (self.z * self.F)
                        - self.N * self.A * self.Da * (Cwaconc - Cwadil) / self.La
                        - self.N * self.A * self.Dc * (Cwcconc - Cwcdil) / self.Lc) / (self.N * self.Vk_conc)

        DdilDt = (self.Qdil * (Cdil_in - Cdil)
                        - self.N * self.phi * j * self.A / (self.z * self.F)
                        + self.N * self.A * self.Da * (Cwaconc - Cwadil) / self.La
                        + self.N * self.A * self.Dc * (Cwcconc - Cwcdil) / self.Lc) / (self.N * self.Vk_dil)

        # at inlet, consider recirculation of the feed stream
        Dconc_inDt = self.Qconc * (Cconc - Cconc_in)/self.VT_conc
        Ddil_inDt = self.Qdil * (Cdil - Cdil_in)/self.VT_dil

        if self.torch:
            return torch.tensor([DconcDt, DdilDt, Dconc_inDt, Ddil_inDt])
        else:
            return [DconcDt, DdilDt, Dconc_inDt, Ddil_inDt]
