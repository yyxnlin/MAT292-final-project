import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fhn(t, y, a, b, tau, I):
    v, w = y
    dv = v - v**3 / 3 - w + I
    dw = (v + a - b*w) / tau
    return [dv, dw]

def simulate_fhn(params, t_eval):
    a, b, tau, I, v0, w0 = params
    sol = solve_ivp(lambda t, y: fhn(t, y, a, b, tau, I),
                    [0, t_eval[-1]], [v0, w0],
                    t_eval=t_eval, method="LSODA")
    return sol.y[0]

def loss(params, t_sub, ecg_sub):
    """
    Least squares used for fitting
    """
    v_sim = simulate_fhn(params, t_sub)
    # Penalize explosion
    if np.any(np.abs(v_sim) > 1e3):
        return 1e12
    return np.sum((v_sim - ecg_sub)**2)


def fit_fhn_to_segment(ecg_sub, t_sub, timeout=10):
    params0 = [1, 1, 1, 0.5, ecg_sub[0], 0.0]
    start_time = time.time()
    try:
        res = minimize(lambda p: loss(p, t_sub, ecg_sub), params0, method="Nelder-Mead")

        # if takes too long, just skip
        if time.time() - start_time > timeout:
            return None
    except Exception:
        return None
    return res








# --- Aliev-Panfilov system with step-function epsilon(u) ---
def aliev_panfilov(t, y, a, k, epsilon_low, epsilon_high):
    u, v = y

    # Step function for epsilon
    eps = epsilon_low if u < a else epsilon_high

    du = k * u * (1 - u) * (u - a) - u * v
    dv = eps * (k * u - v)

    return [du, dv]

# --- Simulation wrapper ---
def simulate_ap(params, t_eval):
    a, k, epsilon_low, epsilon_high, u0, v0 = params

    # Check for invalid parameters
    if k <= 0 or epsilon_low <= 0 or epsilon_high <= 0:
        return np.full_like(t_eval, np.inf)

    try:
        sol = solve_ivp(
            lambda t, y: aliev_panfilov(t, y, a, k, epsilon_low, epsilon_high),
            [0, t_eval[-1]],
            [u0, v0],
            t_eval=t_eval,
            method="RK45",
            max_step=0.01,  # small step for sharp QRS spikes
            rtol=1e-6,
            atol=1e-8
        )
    except Exception:
        return np.full_like(t_eval, np.inf)

    # Ensure solver returns full output
    if sol.y.shape[1] != len(t_eval):
        return np.full_like(t_eval, np.inf)

    return sol.y[0]

# --- Loss function for fitting ---
def loss_ap(params, t_eval, ecg_subsampled):
    u_sim = simulate_ap(params, t_eval)
    return np.sum((u_sim - ecg_subsampled)**2)



def fit_ap_to_segment(ecg_sub, t_sub, timeout=10):
    params0 = [
        0.05,               # a
        5.0,                # k
        1.0,                # epsilon_low
        0.1,                # epsilon_high
        ecg_sub[0],  # u0 (adapted per beat)
        0.0                 # v0
    ]

    bounds = [
        (0.04, 0.07),   # a
        (5.0, 7.0),   # k
        (0.5, 2.0),    # epsilon_low
        (0.05, 0.2),   # epsilon_high
        (0.0, 1.0),    # u0
        (0.0, 0.2)     # v0
    ]
    
    start_time = time.time()
    try:
        res = minimize(lambda p: loss_ap(p, t_sub, ecg_sub), params0, method='Nelder-Mead', bounds=bounds)

        # if takes too long, just skip
        # if time.time() - start_time > timeout:
        #     print("timeout")
        #     return None
    except Exception:
        print("exception")
        return None
    # print("returning res:", res)
    return res