""" A module to translate reduced pressure to pressure and the opposite """

# Imports
# =======
import numpy as np


# Core
# ====

def Pred2P(Pr, PARAM):
    """
    Computes the real pore pressure, using the reduced pore pressure profile.
    Input and output variables are either all adim or all dim.

    - Parameters:
    	+ :param Pr: nd array 1D, reduced pore pressure.
    	+ :param PARAM: dictionnary of parameters.
    - Outputs:
    	+ :return: P, nd array 1D, same shape as Pr, real pore pressure.

    """
    # Unpack
    rho = PARAM['rho']
    alpha = PARAM['alpha']
    g = PARAM['g']
    Nx = PARAM['Nx']
    h = PARAM['h_']
    Z0 = PARAM['Z0_']
    X_scale = PARAM['X_scale']
    Z_scale = PARAM['Z_scale']
    P_scale = PARAM['P_scale']


    X = np.linspace(0, Nx*h, num=Nx+1) #* X_scale/X_scale
    Xref = Z0/np.sin(alpha) * Z_scale/X_scale

    # Core
    P = Pr - rho*g*np.sin(alpha)*(X-Xref) * X_scale/P_scale

    return P
#---------------------------------------------------------------------------------

def P2Pred(P, PARAM):
    """
    Computes the reduced pore pressure, using the real pore pressure profile.
    Input and output variables are either all adim or all dim.

    - Parameters:
    	+ :param P: nd array 1D, real pore pressure.
    	+ :param PARAM: dictionnary of parameters.
    - Outputs:
    	+ :return: Pr, nd array 1D, same shape as P, reduced pore pressure.

    """
    # Unpack
    rho = PARAM['rho']
    alpha = PARAM['alpha']
    g = PARAM['g']
    Nx = PARAM['Nx']
    h = PARAM['h_']
    Z0 = PARAM['Z0_']
    X_scale = PARAM['X_scale']
    Z_scale = PARAM['Z_scale']
    P_scale = PARAM['P_scale']

    X = np.linspace(0, Nx*h, num=Nx+1) #* X_scale/X_scale
    Xref = Z0/np.sin(alpha) * Z_scale/X_scale
    # Core
    Pr = P + rho*g*np.sin(alpha)*(X-Xref) * X_scale/P_scale

    return Pr
