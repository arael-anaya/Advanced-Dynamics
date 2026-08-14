
import numpy as np

L1 = 3 
L2 = 2 
L3 = 1

def getBeta(x , y):
    num = L1**2 + L2**2 - x**2 - y**2
    den = 2 * L1 * L2

    beta = np.arccos(num/den)


    return np.array((beta , -beta))


def getAlpha(x , y):

    num =  x**2 + y**2 + L1**2 - L2**2 
    den = 2 * L1 * np.sqrt(x**2 + y**2)

    alpha = np.arccos(num/den)

    return np.array((alpha, -alpha))


def forward_kinematics(theta1, theta2, theta3):

    def transform(theta, L):
        return np.array([
            [np.cos(theta) , -np.sin(theta) , L*np.cos(theta)],
            [np.sin(theta) , np.cos(theta) , L*np.sin(theta)],
            [0  , 0 ,  1]
        ])

    T1 = transform(theta1 , L1)
    T2 = transform(theta2 , L2)
    T3 = transform(theta3 , L3)

    T = T1 @ T2 @ T3

    x = T[0,2]
    y = T[1,2]
    theta = theta1 + theta2 + theta3

    return np.array([x,y,theta])


def main():

    states = {'A' : [4, 0.5 , 22] , 'B' : [1.5 , -1.5, -105]}

    for name, state in states.items():
        print(f"State {name}: {state}")
        x = state[0]
        y = state[1]
        theta =  np.radians(state[2])

        x_w = x - L3 * np.cos(theta)
        y_w = y - L3 * np.sin(theta)

        gamma = np.arctan2(y_w, x_w)

        alpha = getAlpha(x_w, y_w)
        beta = getBeta(x_w , y_w)

        theta1 = gamma - alpha
        theta2 = np.pi - np.array(beta)
        theta3 = theta - theta1 - theta2

        for sol in range(len(theta1)):

            print(f"Solution {sol+1}:")
            print(f"Theta 1: {theta1[sol]}")
            print(f"Theta 2: {theta2[sol]}")
            print(f"Theta 3: {theta3[sol]}")

            check = forward_kinematics(theta1[sol], theta2[sol], theta3[sol])
            print(f"Solution Validation: {check}")
            



    


if __name__ == "__main__":
    main()

