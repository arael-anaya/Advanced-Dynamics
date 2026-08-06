import numpy as np

# Paramerts
m = 80
r_o = 1
k = 20000
g = 9.81
theta_TD = np.radians(68)
y_o = 1.2
v_o = 3.0

params = {'m' : m , 'k' : k , 'g' : g, 'r_o' : r_o,'r' : 0 , 'theta' : 0}



def touchdownState(y_o , v_o, theta_TD, r_o, g):
    r = r_o
    theta = np.arctan2(r_o * np.sin(theta_TD) , -r_o * np.cos(theta_TD))

    x_TD_dot = v_o
    y_TD_dot = - (2*g*(y_o - r_o * np.sin(theta_TD)))**.5


    r_dot = x_TD_dot * np.cos(theta) + y_TD_dot * np.sin(theta)

    theta_dot = (-x_TD_dot * np.sin(theta) + y_TD_dot * np.cos(theta)) / r_o



    return [r , theta, r_dot , theta_dot]

def stance_RHS(t , s, params):
    r_dot = s[0]
    theta_dot = s[1]


    r_ddot = params['r'] * theta_dot**2 - params['g'] * np.sin(params['theta']) - (params['k'] / params['m'])* (params['r'] - params['r_o'])
    theta_ddot = -(2/params['r']) * r_dot * theta_dot - (params['g'] / params['r']) * np.cos(params['theta'])

    return [r_dot , theta_dot, r_ddot, theta_ddot]


