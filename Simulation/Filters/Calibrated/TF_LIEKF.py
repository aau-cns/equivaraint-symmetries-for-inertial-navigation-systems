# Add main directory to path
import pdb
import sys, os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from Symmetries.Calibrated.SE23_se3.Symmetry import *
from scipy.linalg import expm
from pylie import SE23


class TF_LIEKF:
    def __init__(self, initial_att_noise=1.0, initial_vel_noise=1.0, initial_pos_noise=1.0, initial_bias_noise=0.01, propagationonly=False, equivariant_output=False):
        self.X_hat = State()    # (R,v,p)
        sigma_vec = np.concatenate((
            np.ones((1, 3)) * initial_att_noise**2,
            np.ones((1, 3)) * initial_vel_noise**2,
            np.ones((1, 3)) * initial_pos_noise**2,
            np.ones((1, 6)) * initial_bias_noise**2), axis=1)
        self.Sigma = np.eye(sigma_vec.shape[1]) * sigma_vec
        self.propagation_only = propagationonly
        self.dof = 15

    def stateEstimate(self):
        return self.X_hat

    def getEstimate(self):
        xi_hat = self.stateEstimate()
        R = xi_hat.T.R().as_matrix()
        v = xi_hat.T.x().as_vector()
        p = xi_hat.T.w().as_vector()
        bw = xi_hat.b[0:3, 0:1]
        ba = xi_hat.b[3:6, 0:1]
        return R, p, v, bw, ba, np.zeros((3, 1)), np.zeros((3, 1))

    def update(self, vel : np.ndarray, omega_noise : float, acc_noise : float, tau_noise : float, virtual_noise : float, y : np.ndarray, meas_noise : float, dt : float):

        # Settings
        noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 6)) * tau_noise ** 2), axis=1)
        R = np.eye(noise_vec.shape[1]) * noise_vec
        Q = np.eye(3) * (meas_noise**2)

        # Filter matrices
        u = input_from_vector(vel)
        A0t = self.stateMatrixA_CT(u)
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
                Ct = np.hstack((np.zeros((3, 6)), np.eye(3), np.zeros((3, 6))))
                res = self.X_hat.T.R().inv().as_matrix() @ (y - self.X_hat.T.w().as_vector())
                S = Ct @ self.Sigma @ Ct.T + self.X_hat.T.R().inv().as_matrix() @ Q @ self.X_hat.T.R().as_matrix()
                K = self.Sigma @ Ct.T @ np.linalg.inv(S)
                Delta = K @ res
                Delta_T = SE23.exp(SE23.wedge(Delta[0:9, 0:1]))
                self.X_hat.T = self.X_hat.T * Delta_T
                self.X_hat.b = (blockDiag(SO3LeftJacobian(-Delta[0:3, 0:1]), SO3LeftJacobian(-Delta[0:3, 0:1])) @ Delta[9:15, 0:1] +
                                blockDiag(Delta_T.R().inv().as_matrix(), Delta_T.R().inv().as_matrix()) @ self.X_hat.b)
                self.Sigma = (np.eye(15) - K @ Ct) @ self.Sigma

        return False

    def stateMatrixA_CT(self, u : InputSpace) -> np.ndarray:
        A0t = np.zeros((15, 15))
        wb = np.vstack((u.as_wa_vec() - self.X_hat.b, np.zeros((3, 1))))
        bw = self.X_hat.b[0:3]
        ba = self.X_hat.b[3:6]
        A0t[0:9, 0:9] = -SE23.adjoint(wb)
        A0t[6:9, 3:6] = np.eye(3)
        A0t[0:6, 9:15] = -np.eye(6)
        A0t[0:6, 0:3] -= np.vstack((SO3.wedge(bw), SO3.wedge(ba)))
        A0t[9:12, :] = SO3.wedge(bw) @ A0t[0:3, :]
        A0t[12:15, :] = SO3.wedge(ba) @ A0t[0:3, :]
        return A0t

    def inputMatrixBt_CT(self) -> np.ndarray:
        tmp = np.vstack((-np.eye(6), np.zeros((3, 6))))
        Bt = blockDiag(tmp, np.eye(6))
        bw = self.X_hat.b[0:3]
        ba = self.X_hat.b[3:6]
        Bt[9:12, :] = SO3.wedge(bw) @ Bt[0:3, :]
        Bt[12:15, :] = SO3.wedge(ba) @ Bt[0:3, :]
        return Bt

    def computeNEES(self, xi_data):
        xi = stateFromData(xi_data)

        R_hat = self.X_hat.T.R().as_matrix()
        R = xi.T.R().as_matrix()
        e_R = SO3.log(SO3.from_matrix(np.linalg.inv(R_hat) @ R))

        x_hat = np.vstack((self.X_hat.T.x().as_vector(), self.X_hat.T.w().as_vector()))
        x = np.vstack((xi.T.x().as_vector(), xi.T.w().as_vector()))
        e_x = blockDiag(np.linalg.inv(R_hat), np.linalg.inv(R_hat)) @ (x - x_hat)

        e_b = xi.b - blockDiag(np.linalg.inv(np.linalg.inv(R_hat) @ R), np.linalg.inv(np.linalg.inv(R_hat) @ R)) @ self.X_hat.b

        eps = np.vstack((e_R, e_x, e_b))

        ness = eps.T @ np.linalg.inv(self.Sigma) @ eps
        nav_ness = eps[0:9, :].T @ np.linalg.inv(self.Sigma[0:9, 0:9]) @ eps[0:9, :]
        bias_ness = eps[9:self.dof, :].T @ np.linalg.inv(self.Sigma[9:self.dof, 9:self.dof]) @ eps[9:self.dof, :]

        return float(ness), float(nav_ness), float(bias_ness)
