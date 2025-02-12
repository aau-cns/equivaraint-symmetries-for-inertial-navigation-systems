# Add main directory to path
import pdb
import sys, os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from Symmetries.Calibration.SE23_se3.Symmetry import *
from scipy.linalg import expm
from pylie import SE23


class cal_IEKF:
    def __init__(self, initial_nav_noise=1.0, initial_bias_noise=0.01, propagationonly=False):
        self.X_hat = State()    # (R,v,p,b,t)

        sigma_vec = np.concatenate((np.ones((1, 9)) * initial_nav_noise ** 2, np.ones((1, 6)) * initial_bias_noise ** 2, np.ones((1, 3)) * (initial_nav_noise / 5) ** 2), axis=1)

        self.Sigma = np.eye(sigma_vec.shape[1]) * sigma_vec
        self.propagation_only = propagationonly
        self.dof = 18
    
    def stateEstimateMatrix(self):
        T = np.vstack((np.hstack((self.X_hat.T.R().as_matrix(), self.X_hat.T.x().as_vector(), self.X_hat.T.w().as_vector())), np.hstack((np.zeros((2, 3)), np.eye(2)))))
        return T, self.X_hat.b, self.X_hat.t

    def stateEstimate(self):
        return self.X_hat

    def getEstimate(self):
        xi_hat = self.stateEstimate()
        R = xi_hat.T.R().as_matrix()
        v = xi_hat.T.x().as_vector()
        p = xi_hat.T.w().as_vector()
        bw = xi_hat.b[0:3, 0:1]
        ba = xi_hat.b[3:6, 0:1]
        t = xi_hat.t
        return R, p, v, bw, ba, np.zeros((3, 1)), t

    def update(self, vel : np.ndarray, omega_noise : float, acc_noise : float, tau_noise : float,  virtual_noise : float, y : np.ndarray, meas_noise : float, dt : float):

        # Settings
        noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 6)) * tau_noise ** 2), axis=1)
        R = np.eye(noise_vec.shape[1]) * noise_vec
        Q = np.eye(3) * (meas_noise**2)

        # Filter matrices
        u = input_from_vector(vel)
        A0t = self.stateMatrixA_CT()
        Bt = self.inputMatrixBt_CT()

        # Propagation
        M = Bt @ R @ Bt.T

        try:
            Phi_DT = expm(A0t*dt)
        except ValueError as e:
            print ('Phi_DT = expm(A0t*dt) Error')
            return True

        g = np.zeros((3, 1))
        g[2] = -9.81
        R_k = self.X_hat.T.R().as_matrix()
        v_k = self.X_hat.T.x().as_vector()
        p_k = self.X_hat.T.w().as_vector()
        R_kk = R_k @ SO3.exp((SO3.vee(u.w) - self.X_hat.b[0:3]) * dt).as_matrix()
        v_kk = v_k + (R_k @ (u.a - self.X_hat.b[3:6]) + g) * dt
        p_kk = p_k + v_k * dt + 0.5 * (R_k @ (u.a - self.X_hat.b[3:6]) + g) * (dt ** 2)
        T_kk = np.vstack((np.hstack((R_kk, v_kk, p_kk)), np.hstack((np.zeros((2, 3)), np.eye(2)))))
        self.X_hat.T = SE23.from_matrix(T_kk)
        self.Sigma = Phi_DT @ self.Sigma @ Phi_DT.T + (M * dt)

        # Update when measurement available and if allowed
        if not self.propagation_only:
            if not np.isnan(y[0, 0:1]):
                # Ct = self.outputMatrixC()
                # res = y - ((self.X_hat.T.R().as_matrix() @ self.X_hat.t) + self.X_hat.T.w().as_vector())
                Ct = self.outputMatrixCTrick(y)
                res = ((self.X_hat.T.R().as_matrix() @ self.X_hat.t) + self.X_hat.T.w().as_vector()) - y
                S = Ct @ self.Sigma @ Ct.T + Q
                K = self.Sigma @ Ct.T @ np.linalg.inv(S)
                Delta = K @ res
                Delta_T = SE23.exp(SE23.wedge(Delta[0:9, 0:1])).as_matrix()
                Delta_b = Delta[9:15, 0:1]
                Delta_t = Delta[15:18, 0:1]
                self.X_hat.T = SE23.from_matrix(Delta_T) * self.X_hat.T
                self.X_hat.b = Delta_b + self.X_hat.b
                self.X_hat.t = Delta_t + self.X_hat.t
                self.Sigma = (np.eye(self.dof) - K @ Ct) @ self.Sigma

        return False

    def stateMatrixA_CT(self) -> np.ndarray:
        A0t = np.zeros((18, 18))
        A0t[0:9, 0:9] = np.hstack((blockDiag(np.vstack((np.zeros((3, 3)), SO3.skew(g))), np.eye(3)), np.zeros((9, 3))))
        tmp = -self.X_hat.T.Adjoint()
        A0t[0:9, 9:15] = tmp[0:9, 0:6]
        return A0t

    def inputMatrixBt_CT(self) -> np.ndarray:
        Bt = np.zeros((18, 12))
        tmp = self.X_hat.T.Adjoint()
        Bt[0:15, :] = blockDiag(tmp[0:9, 0:6], np.eye(6))
        return Bt

    def outputMatrixC(self) -> np.ndarray:
        C0 = np.hstack((-SO3.skew((self.X_hat.T.R().as_matrix() @ self.X_hat.t) + self.X_hat.T.w().as_vector()),
                        np.zeros((3, 3)),
                        np.eye(3),
                        np.zeros((3, 6)),
                        self.X_hat.T.R().as_matrix()))
        return C0

    def outputMatrixCTrick(self, y : np.ndarray) -> np.ndarray:
        C0 = np.hstack((SO3.wedge(y),
                        np.zeros((3, 3)),
                        -np.eye(3),
                        np.zeros((3, 6)),
                        -self.X_hat.T.R().as_matrix()))
        return C0

    def computeError(self, xi_data) -> float:
        xi = stateFromData(xi_data)
        T_hat, b_hat, t_hat = self.stateEstimateMatrix()
        T = np.vstack((np.hstack((xi.T.R().as_matrix(), xi.T.x().as_vector(), xi.T.w().as_vector())), np.hstack((np.zeros((2, 3)), np.eye(2)))))
        e_T = SE23.log(SE23.from_matrix(T @ np.linalg.inv(T_hat)))
        e_b = xi.b - b_hat
        e_t = xi.t - t_hat
        eps = np.vstack((e_T, e_b, e_t))
        err = eps.T @ np.linalg.inv(self.Sigma) @ eps
        return float(err)
