# Add main directory to path
import pdb
import sys, os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from Symmetries.Calibrated.SO3_R12.System import *
from scipy.linalg import expm
from pylie import SE23

###############
# NOT WORKING #
###############

class Weights:
    def __init__(self, n, k, alpha):
        self.n = n

        self.l = (alpha**2) * (self.n + k) - self.n

        self.wm0 = (self.l / (self.n + self.l))
        self.wmi = 1 / (2 * (self.n + self.l))
        self.wc = (self.l / (self.n + self.l)) + (3 - alpha**2)

class UKF:
    def __init__(self, initial_att_noise=1.0, initial_vel_noise=1.0, initial_pos_noise=1.0, initial_bias_noise=0.01, propagationonly=False):
        self.X_hat = State()    # (R,v,p,b)
        sigma_vec = np.concatenate((
            np.ones((1, 3)) * initial_att_noise**2,
            np.ones((1, 3)) * initial_vel_noise**2,
            np.ones((1, 3)) * initial_pos_noise**2,
            np.ones((1, 6)) * initial_bias_noise**2), axis=1)
        self.Sigma = np.eye(sigma_vec.shape[1]) * sigma_vec
        self.propagation_only = propagationonly
        self.dof = 15

        self.k = 2
        self.alpha = 1.0e-1

        self.state_weights = Weights(self.dof, self.k, self.alpha)
        self.covariance_weights = Weights(12, self.k, self.alpha)

    def stateEstimate(self):
        return self.X_hat

    def getEstimate(self):
        xi_hat = self.stateEstimate()
        R = xi_hat.R.as_matrix()
        v = xi_hat.v
        p = xi_hat.p
        bw = xi_hat.b[0:3, 0:1]
        ba = xi_hat.b[3:6, 0:1]
        return R, p, v, bw, ba, np.zeros((3, 1)), np.zeros((3, 1))

    def getSigmaPointsPropagation(self, w, S, u, dt):
        s = np.sqrt(w.l + w.n) * np.linalg.cholesky(S).T
        Xi = np.zeros((2 * w.n, self.dof))
        for i in range(w.n):
            if s[:, i].shape[0] < self.dof:
                x = np.hstack((s[0:6, i], np.zeros((3)), s[6:12, i])).reshape(15, 1)
            else:
                x = s[:, i].reshape(15, 1)
            Xi[i, :] = State.log(self.f(self.X_hat * State.exp(x), u, dt)).T
            Xi[i + w.n, :] = State.log(self.f(self.X_hat * State.exp(-x), u, dt)).T
        return Xi

    def getSigmaPointsUpdate(self, w, S):
        s = np.sqrt(w.l + w.n) * np.linalg.cholesky(S).T
        Xi = np.zeros((2 * self.dof, 3))
        for i in range(self.dof):
            Xi[i, :] = (self.X_hat * State.exp(s[:, i].reshape(15, 1))).p.T
            Xi[i + self.dof, :] = (self.X_hat * State.exp(-s[:, i].reshape(15, 1))).p.T
        return Xi, s

    def f(self, X : State, u : InputSpace, dt : float):
        g = np.zeros((3, 1))
        g[2] = -9.81
        R_k = X.R.as_matrix()
        v_k = X.v
        p_k = X.p
        R_kk = R_k @ SO3.exp((SO3.vee(u.w) - X.b[0:3]) * dt).as_matrix()
        v_kk = v_k + (R_k @ (u.a - X.b[3:6]) + g) * dt
        p_kk = p_k + v_k * dt + 0.5 * (R_k @ (u.a - X.b[3:6]) + g) * (dt ** 2)
        T_kk = np.vstack((np.hstack((R_kk, v_kk, p_kk)), np.hstack((np.zeros((2, 3)), np.eye(2)))))
        X.T = SE23.from_matrix(T_kk)
        return X

    def update(self, vel : np.ndarray, omega_noise : float, acc_noise : float, tau_noise : float,  virtual_noise : float, y : np.ndarray, meas_noise : float, dt : float):

        # Settings
        noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 6)) * tau_noise ** 2), axis=1)
        Q = np.eye(noise_vec.shape[1]) * noise_vec
        R = np.eye(3) * (meas_noise**2)

        # Filter input
        u = input_from_vector(vel)

        # Mean Porpoagation
        self.X_hat = self.f(self.X_hat, u, dt)

        # Get sigma points for state covariance
        Xi = self.getSigmaPointsPropagation(self.state_weights, self.Sigma, u, dt)
        X_ = self.state_weights.wmi * np.sum(Xi, 0)
        Xi -= X_
        Sigma = self.state_weights.wmi * Xi.T.dot(Xi) + self.state_weights.wc * np.outer(X_, X_)

        # Get sigma points for noise covariance
        Xi = self.getSigmaPointsPropagation(self.covariance_weights, Q, u, dt)
        X_ = self.covariance_weights.wmi * np.sum(Xi, 0)
        Xi -= X_
        Q = self.covariance_weights.wmi * Xi.T.dot(Xi) + self.covariance_weights.wc * np.outer(X_, X_)

        self.Sigma = Sigma + Q

        # Update when measurement available and if allowed
        if not self.propagation_only:
            if not np.isnan(y[0, 0:1]):

                # Get sigma points for update
                Yi, s = self.getSigmaPointsUpdate(self.state_weights, self.Sigma)
                Y_ = self.state_weights.wm0 * self.X_hat.p.T + self.state_weights.wmi * np.sum(Yi, 0)
                Yi -= Y_
                Yhat = self.X_hat.p.T - Y_

                R += self.state_weights.wmi * Yi.T.dot(Yi) + self.state_weights.wc * np.outer(Yhat, Yhat)
                Rc = self.state_weights.wc * np.hstack([s.T, -s.T]).dot(Yi)

                K = np.linalg.solve(R, Rc.T).T
                Delta = K @ (y - Y_.T)

                self.X_hat.R = SO3.from_matrix(SO3.exp(Delta[0:3, :]).as_matrix() @ self.X_hat.R.as_matrix())
                self.X_hat.v = Delta[3:6, :] + self.X_hat.v
                self.X_hat.p = Delta[6:9, :] + self.X_hat.p
                self.X_hat.b = Delta[9:15, :] + self.X_hat.b
                self.Sigma -= K.dot(R).dot(K.T)
                self.Sigma = (self.Sigma + self.Sigma.T) / 2

        return False

    def computeNEES(self, xi_data):
        xi = stateFromData(xi_data)
        R_hat = self.X_hat.R.as_matrix()
        R = xi.R.as_matrix()
        e_R = SO3.log(SO3.from_matrix(R @ np.linalg.inv(R_hat)))
        e_v = xi.v - self.X_hat.v
        e_p = xi.p - self.X_hat.p
        e_b = xi.b - self.X_hat.b
        eps = np.vstack((e_R, e_v, e_p, e_b))
        ness = eps.T @ np.linalg.inv(self.Sigma) @ eps
        nav_ness = eps[0:9, :].T @ np.linalg.inv(self.Sigma[0:9, 0:9]) @ eps[0:9, :]
        bias_ness = eps[9:self.dof, :].T @ np.linalg.inv(self.Sigma[9:self.dof, 9:self.dof]) @ eps[9:self.dof, :]
        return float(ness), float(nav_ness), float(bias_ness)
