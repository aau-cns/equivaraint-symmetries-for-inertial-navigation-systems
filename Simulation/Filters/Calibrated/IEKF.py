# Add main directory to path
import pdb
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from Symmetries.Calibrated.SE23_se3.Symmetry import *
from scipy.linalg import expm
from pylie import SE23


class IEKF:
    def __init__(self, initial_att_noise=1.0, initial_vel_noise=1.0, initial_pos_noise=1.0, initial_bias_noise=0.01, propagationonly=False, equivariant_output=False):
        self.X_hat = State()    # (R,v,p,b)
        sigma_vec = np.concatenate((
            np.ones((1, 3)) * initial_att_noise**2,
            np.ones((1, 3)) * initial_vel_noise**2,
            np.ones((1, 3)) * initial_pos_noise**2,
            np.ones((1, 6)) * initial_bias_noise**2), axis=1)
        self.Sigma = np.eye(sigma_vec.shape[1]) * sigma_vec
        self.propagation_only = propagationonly
        self.dof = 15
        self.t = None
        self.u = None
        self.equivariant_output = equivariant_output
        if self.equivariant_output:
            print("Equivariant output enabled")
        self.perform_left_right_comparison = False

    def stateEstimateMatrix(self):
        T = np.vstack((np.hstack((self.X_hat.T.R().as_matrix(), self.X_hat.T.x().as_vector(), self.X_hat.T.w().as_vector())), np.hstack((np.zeros((2, 3)), np.eye(2)))))
        return T, self.X_hat.b

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

    def propagate(self, t: float, vel: np.ndarray, omega_noise: float, acc_noise: float, tau_noise: float,
                  virtual_noise: float):
        # Time
        if self.t is None:
            self.t = t
            self.u = input_from_vector(vel)
            return True

        dt = t - self.t

        if dt < 1.0e-6:
            return True
        if dt > 0.1:
            return True

        # Settings
        noise_vec = np.concatenate(
            (np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 6)) * tau_noise ** 2),
            axis=1)
        R = np.eye(noise_vec.shape[1]) * noise_vec

        # Filter matrices
        A0t = self.stateMatrixA_CT()
        Bt = self.inputMatrixBt_CT()

        # Propagation
        M = Bt @ R @ Bt.T

        try:
            Phi_DT = expm(A0t * dt)
        except ValueError as e:
            print('Phi_DT = expm(A0t*dt) Error')
            return True

        g = np.zeros((3, 1))
        g[2] = -9.81
        R_k = self.X_hat.T.R().as_matrix()
        v_k = self.X_hat.T.x().as_vector()
        p_k = self.X_hat.T.w().as_vector()
        R_kk = R_k @ SO3.exp((SO3.vee(self.u.w) - self.X_hat.b[0:3]) * dt).as_matrix()
        v_kk = v_k + (R_k @ (self.u.a - self.X_hat.b[3:6]) + g) * dt
        p_kk = p_k + v_k * dt + 0.5 * (R_k @ (self.u.a - self.X_hat.b[3:6]) + g) * (dt ** 2)
        T_kk = np.vstack((np.hstack((R_kk, v_kk, p_kk)), np.hstack((np.zeros((2, 3)), np.eye(2)))))
        self.X_hat.T = SE23.from_matrix(T_kk)
        self.Sigma = Phi_DT @ self.Sigma @ Phi_DT.T + (M * dt)

        self.t = t
        self.u = input_from_vector(vel)

        return False

    # def update(self, y: np.ndarray, meas_noise: float):
    #
    #     Q = np.eye(3) * (meas_noise ** 2)
    #
    #     # Update when measurement available and if allowed
    #     if not self.propagation_only:
    #         if not np.isnan(y[0, 0:1]):
    #             if self.equivariant_output:
    #                 ## Right invariant (equivariant) model [R^T(pi - p)]
    #                 Ct = np.hstack((0.5 * SO3.skew(y + self.X_hat.T.w().as_vector()), np.zeros((3, 3)), -np.eye(3),
    #                                 np.zeros((3, 6))))
    #                 # Ct = np.hstack((SO3.skew(y), np.zeros((3, 3)), -np.eye(3), np.zeros((3, 6))))
    #                 res = self.X_hat.T.w().as_vector() - y
    #             else:
    #                 Ct = np.hstack(
    #                     (SO3.skew(-self.X_hat.T.w().as_vector()), np.zeros((3, 3)), np.eye(3), np.zeros((3, 6))))
    #                 res = y - self.X_hat.T.w().as_vector()
    #             S = Ct @ self.Sigma @ Ct.T + Q
    #             S_inv = np.linalg.inv(S)
    #             nis = res.T @ S_inv @ res
    #             K = self.Sigma @ Ct.T @ S_inv
    #             Delta = K @ res
    #
    #             if self.perform_left_right_comparison:
    #                 ### Left update
    #                 Left_Ct = np.hstack((np.zeros((3, 6)), np.eye(3), np.zeros((3, 6))))
    #                 Left_res = self.X_hat.T.R().inv().as_matrix() @ (y - self.X_hat.T.w().as_vector())
    #                 Left_Sigma = blockDiag(self.X_hat.T.inv().Adjoint(), np.eye(6)) @ self.Sigma @ blockDiag(
    #                     self.X_hat.T.inv().Adjoint(), np.eye(6)).T
    #                 Left_S = Left_Ct @ Left_Sigma @ Left_Ct.T + self.X_hat.T.R().inv().as_matrix() @ Q @ self.X_hat.T.R().as_matrix()
    #                 Left_K = Left_Sigma @ Left_Ct.T @ np.linalg.inv(Left_S)
    #                 Left_Delta = Left_K @ Left_res
    #                 ###
    #
    #                 ### Comparison
    #                 RightDeltaT = self.X_hat.T.Adjoint() @ Left_Delta[0:9, :]
    #                 print(RightDeltaT - Delta[0:9, :])
    #                 ###
    #
    #                 ### Left and right Delta are exactly the same ###
    #
    #             Delta_T = SE23.exp(SE23.wedge(Delta[0:9, 0:1])).as_matrix()
    #             Delta_b = Delta[9:15, 0:1]
    #             self.X_hat.T = SE23.from_matrix(Delta_T) * self.X_hat.T
    #             self.X_hat.b = Delta_b + self.X_hat.b
    #             self.Sigma = (np.eye(15) - K @ Ct) @ self.Sigma
    #
    #             # RESET
    #             exp_Gamma = blockDiag(expm(0.5 * SE23.adjoint(Delta[0:9, 0:1])), np.eye(6))
    #             self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
    #
    #             return nis


    def update(self, vel : np.ndarray, omega_noise : float, acc_noise : float, tau_noise : float, virtual_noise : float, y : np.ndarray, meas_noise : float, dt : float, propagate : bool = True):

        # Settings
        Q = np.eye(3) * (meas_noise**2)

        if propagate:
            noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 6)) * tau_noise ** 2), axis=1)
            R = np.eye(noise_vec.shape[1]) * noise_vec

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
                if self.equivariant_output:
                    ## Right invariant (equivariant) model [R^T(pi - p)]
                    Ct = np.hstack((0.5 * SO3.skew(y + self.X_hat.T.w().as_vector()), np.zeros((3, 3)), -np.eye(3), np.zeros((3, 6))))
                    # Ct = np.hstack((SO3.skew(y), np.zeros((3, 3)), -np.eye(3), np.zeros((3, 6))))
                    res = self.X_hat.T.w().as_vector() - y
                else:
                    Ct = np.hstack((SO3.skew(-self.X_hat.T.w().as_vector()), np.zeros((3, 3)), np.eye(3), np.zeros((3, 6))))
                    res = y - self.X_hat.T.w().as_vector()
                S = Ct @ self.Sigma @ Ct.T + Q
                S_inv = np.linalg.inv(S)
                nis = res.T @ S_inv @ res
                K = self.Sigma @ Ct.T @ S_inv
                Delta = K @ res

                if self.perform_left_right_comparison:
                    ### Left update
                    Left_Ct = np.hstack((np.zeros((3, 6)), np.eye(3), np.zeros((3, 6))))
                    Left_res = self.X_hat.T.R().inv().as_matrix() @ (y - self.X_hat.T.w().as_vector())
                    Left_Sigma = blockDiag(self.X_hat.T.inv().Adjoint(), np.eye(6)) @ self.Sigma @ blockDiag(self.X_hat.T.inv().Adjoint(), np.eye(6)).T
                    Left_S = Left_Ct @ Left_Sigma @ Left_Ct.T + self.X_hat.T.R().inv().as_matrix() @ Q @ self.X_hat.T.R().as_matrix()
                    Left_K = Left_Sigma @ Left_Ct.T @ np.linalg.inv(Left_S)
                    Left_Delta = Left_K @ Left_res
                    ###

                    ### Comparison
                    RightDeltaT = self.X_hat.T.Adjoint() @ Left_Delta[0:9, :]
                    print(RightDeltaT - Delta[0:9, :])
                    ###

                    ### Left and right Delta are exactly the same ###

                Delta_T = SE23.exp(SE23.wedge(Delta[0:9, 0:1])).as_matrix()
                Delta_b = Delta[9:15, 0:1]
                self.X_hat.T = SE23.from_matrix(Delta_T) * self.X_hat.T
                self.X_hat.b = Delta_b + self.X_hat.b
                self.Sigma = (np.eye(15) - K @ Ct) @ self.Sigma

                # RESET
                exp_Gamma = blockDiag(expm(0.5 * SE23.adjoint(Delta[0:9, 0:1])), np.eye(6))
                self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T

                return nis

    def stateMatrixA_CT(self) -> np.ndarray:
        A0t = np.zeros((15, 15))
        A0t[0:9, 0:9] = np.hstack((blockDiag(np.vstack((np.zeros((3, 3)), SO3.skew(g))), np.eye(3)), np.zeros((9, 3))))
        tmp = -self.X_hat.T.Adjoint()
        A0t[0:9, 9:15] = tmp[0:9, 0:6]
        return A0t

    def inputMatrixBt_CT(self) -> np.ndarray:
        tmp = self.X_hat.T.Adjoint()
        Bt = blockDiag(tmp[0:9, 0:6], np.eye(6))
        return Bt

    def outputMatrixC(self, y: np.ndarray) -> np.ndarray:
        C = np.hstack((SO3.skew(y), np.zeros((3, 3)), -np.eye(3), np.zeros((3, 6))))
        return C

    def computeNEES(self, xi_data):
        xi = stateFromData(xi_data)
        T_hat, b_hat = self.stateEstimateMatrix()
        T = np.vstack((np.hstack((xi.T.R().as_matrix(), xi.T.x().as_vector(), xi.T.w().as_vector())), np.hstack((np.zeros((2, 3)), np.eye(2)))))
        b = xi.b
        e_T = SE23.log(SE23.from_matrix(T @ np.linalg.inv(T_hat)))
        e_b = b - b_hat
        eps = np.vstack((e_T, e_b))
        nees = eps.T @ np.linalg.inv(self.Sigma) @ eps
        nav_ness = e_T.T @ np.linalg.inv(self.Sigma[0:9, 0:9]) @ e_T
        bias_ness = e_b.T @ np.linalg.inv(self.Sigma[9:self.dof, 9:self.dof]) @ e_b
        return float(nees), float(nav_ness), float(bias_ness)
