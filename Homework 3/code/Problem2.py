import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m1 = m2 = 1.0
L1 = L2 = 1.0
g = 9.81

def deriv(t, y):
    t1, t2, w1, w2 = y
    c2, s2 = np.cos(t2), np.sin(t2)
    M = np.array([
        [(m1+m2)*L1**2 + m2*L2**2 + 2*m2*L1*L2*c2, m2*L2**2 + m2*L1*L2*c2],
        [m2*L2**2 + m2*L1*L2*c2,                   m2*L2**2]
    ])
    rhs = np.array([
        m2*L1*L2*s2*w2*(2*w1+w2) - (m1+m2)*g*L1*np.sin(t1) - m2*g*L2*np.sin(t1+t2),
        -m2*L1*L2*s2*w1**2 - m2*g*L2*np.sin(t1+t2)
    ])
    a1, a2 = np.linalg.solve(M, rhs)
    return [w1, w2, a1, a2]

t_end = 25.0
t_eval = np.linspace(0, t_end, 6000)

def run(t1_0, t2_0=0.0, w1_0=0.0, w2_0=0.0):
    return solve_ivp(deriv, [0, t_end], [t1_0, t2_0, w1_0, w2_0],
                     t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-11)

# Part (b)
s = run(np.pi/2)
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax[0].plot(s.t, s.y[0]); ax[0].set_ylabel(r'$\theta_1$ (rad)')
ax[1].plot(s.t, s.y[1]); ax[1].set_ylabel(r'$\theta_2$ (rad)'); ax[1].set_xlabel('t (s)')
ax[0].set_title('Part (b): released from rest at theta1=pi/2, theta2=0')
fig.tight_layout(); fig.savefig('partb.png', dpi=140)

# Part (c)
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6.5))
for d in [-0.1, 0.0, 0.1, 0.2]:
    s = run(np.pi/2 + d)
    ax[0].plot(s.t, s.y[0], label=f'theta1(0)=pi/2{d:+.1f}')
    ax[1].plot(s.t, s.y[1])
ax[0].set_ylabel(r'$\theta_1$ (rad)'); ax[0].legend(fontsize=8)
ax[1].set_ylabel(r'$\theta_2$ (rad)'); ax[1].set_xlabel('t (s)')
ax[0].set_title('Part (c): sensitivity to theta1(0)')
fig.tight_layout(); fig.savefig('partc.png', dpi=140)

plt.show()