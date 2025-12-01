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

