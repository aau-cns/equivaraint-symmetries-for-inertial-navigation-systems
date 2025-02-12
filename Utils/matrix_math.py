import numpy as np

def numericalDifferential(f, x) -> np.ndarray:
    if isinstance(x, float):
        x = np.reshape([x], (1,1))
    h = 1e-6
    fx = f(x)
    n = fx.shape[0]
    m = x.shape[0]
    Df = np.zeros((n, m))
    for j in range(m):
        ej = np.zeros((m,1))
        ej[j,0] = 1.0
        Df[:,j:j+1] = (f(x + h * ej) - f(x - h * ej)) / (2*h)
    return Df

def blockDiag(A : np.ndarray, B : np.ndarray) -> np.ndarray:
    return np.block([[A, np.zeros((A.shape[0], B.shape[1]))],[np.zeros((B.shape[0], A.shape[1])), B]])
