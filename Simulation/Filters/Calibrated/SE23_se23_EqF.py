# Add main directory to path
import pdb
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from Symmetries.Calibrated.SE23_se23.Symmetry import *
from scipy.linalg import expm
import unittest
from pylie import SE23

def continuous_lift(xi : State, U : InputSpace) -> np.ndarray:
    L = np.zeros((18, 1))
    L[0:9, 0:1] = (U.as_W_vec() - xi.b) + SE23.vee(xi.T.inv().as_matrix() @ (G + f_10(xi.T.as_matrix())))
    L[9:18, 0:1] = SE23.adjoint(xi.b) @ L[0:9, 0:1] - U.tau
    return L


class SE23_se23_EqF:
    def __init__(self, initial_att_noise=1.0, initial_vel_noise=1.0, initial_pos_noise=1.0, initial_bias_noise=0.01, propagationonly=False, equivariant_output=False, curvature_correction=False, measure_b_mu=False):
        self.X_hat = SymGroup()
        sigma_vec = np.concatenate((
            np.ones((1, 3)) * initial_att_noise**2,
            np.ones((1, 3)) * initial_vel_noise**2,
            np.ones((1, 3)) * initial_pos_noise**2,
            np.ones((1, 9)) * initial_bias_noise**2), axis=1)
        self.Sigma = np.eye(sigma_vec.shape[1]) * sigma_vec
        self.Dphi0 = stateActionDiff(xi_0)              # (D\theta) * (D\phi_{xi_0}(E) in E = Id)
        self.InnovationLift = np.linalg.pinv(self.Dphi0)
        self.propagation_only = propagationonly
        self.measure_b_mu = measure_b_mu
        self.curvature_correction = curvature_correction
        self.dof = 18
        self.t = None
        self.u = None
        self.equivariant_output = equivariant_output
        if self.curvature_correction:
            print("Curvature correction enabled")
        if self.equivariant_output:
            print("Equivariant output enabled")
        if self.measure_b_mu:
            print("Measuring b_mu")
        else:
            print("Forcing mu = b_mu")
        print(f"Exponential coordinates = {exponential_coords}")
        print(f"Fake exponential = {fake_exponential}")
    
    def stateEstimate(self):
        return stateAction(self.X_hat, xi_0)

    def getEstimate(self):
        xi_hat = self.stateEstimate()
        R = xi_hat.T.R().as_matrix()    # R
        v = xi_hat.T.x().as_vector()    # v
        p = xi_hat.T.w().as_vector()    # p
        bw = xi_hat.b[0:3, 0:1]
        ba = xi_hat.b[3:6, 0:1]
        bmu = xi_hat.b[6:9, 0:1]
        return R, p, v, bw, ba, bmu, np.zeros((3, 1))

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
            (np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 3)) * virtual_noise ** 2, np.ones((1, 9)) * tau_noise ** 2),
            axis=1)
        R = np.eye(noise_vec.shape[1]) * noise_vec

        # Filter matrices
        xi_hat = self.stateEstimate()  # phi_{xi_0}(\hat{X})

        # If not measureing bnu set nu equal bnu, forcing correct dynamics through input
        if not self.measure_b_mu:
            self.u.mu = xi_hat.b[6:9, 0:1]
            # print(f"nu: {u.mu.T}")

        lift = continuous_lift(xi_hat, self.u)
        A0t = self.stateMatrixA_CT(self.u)
        Bt = self.inputMatrixBt_CT(self.u)

        # Propagation
        M = Bt @ R @ Bt.T

        Phi_DT = expm(A0t * dt)
        self.X_hat = self.X_hat * SymGroup.exp(lift * dt)
        self.Sigma = Phi_DT @ self.Sigma @ Phi_DT.T + M * dt

        self.t = t
        self.u = input_from_vector(vel)

        return False

    # def update(self, y: np.ndarray, meas_noise: float):
    #     Q = np.eye(3) * (meas_noise ** 2)
    #     xi_hat = self.stateEstimate()
    #
    #     if not self.propagation_only:
    #         if not np.isnan(y[0, 0:1]):
    #             xi_hat = self.stateEstimate()
    #
    #             if self.measure_b_mu:
    #                 Ct = np.zeros((6, 18))
    #                 Ct[3:6, 9:12] = self.X_hat.B.R().inv().as_matrix() @ SO3.wedge(self.X_hat.B.w().as_vector())
    #                 Ct[3:6, 15:18] = -self.X_hat.B.R().inv().as_matrix()
    #             else:
    #                 Ct = np.zeros((3, 18))
    #
    #             if self.equivariant_output:
    #                 Ct[0:3, 0:3] = 0.5 * SO3.wedge(y + self.X_hat.B.w().as_vector())
    #                 # Ct[0:3, 0:3] = SO3.wedge(y)
    #                 Ct[0:3, 6:9] = -np.eye(3)
    #                 if self.measure_b_mu:
    #                     delta = np.vstack((self.X_hat.B.w().as_vector() - y, np.zeros((3, 1)) - xi_hat.b[6:9, 0:1]))
    #                 else:
    #                     delta = self.X_hat.B.w().as_vector() - y  # rho_hat{X^{-1}}(0)-y
    #             else:
    #                 Ct[0:3, 0:3] = -SO3.wedge(xi_hat.T.w().as_vector())
    #                 Ct[0:3, 6:9] = np.eye(3)
    #                 if self.measure_b_mu:
    #                     delta = np.vstack((y, np.zeros((3, 1)))) - measurePosAndBias(xi_hat)
    #                 else:
    #                     delta = y - measurePos(xi_hat)
    #
    #             S = Ct @ self.Sigma @ Ct.T + Q
    #             Sinv = np.linalg.inv(S)
    #             nis = delta.T @ Sinv @ delta
    #             K = self.Sigma @ Ct.T @ Sinv
    #             Delta = self.InnovationLift @ K @ delta
    #             self.X_hat = SymGroup.exp(Delta) * self.X_hat
    #             self.Sigma = (np.eye(18) - K @ Ct) @ self.Sigma
    #             if self.curvature_correction:
    #                 Gamma = 0.5 * grp_adj(K @ delta)
    #                 exp_Gamma = expm(Gamma)
    #                 self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
    #             return nis

    def update(self, vel : np.ndarray, omega_noise : float, acc_noise : float, tau_noise : float, virtual_noise : float, y : np.ndarray, meas_noise : float, dt : float, propagate : bool = True):

        # Settings
        Q = np.eye(3) * (meas_noise**2)
        if self.measure_b_mu:
            Q = blockDiag(Q, np.eye(3) * 1e-3)

        xi_hat = self.stateEstimate()       #phi_{xi_0}(\hat{X})

        if propagate:
            noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 3)) * virtual_noise ** 2, np.ones((1, 9)) * (tau_noise) ** 2), axis=1)
            R = np.eye(noise_vec.shape[1]) * noise_vec

            # Input (w, a, mu, tau)
            u = input_from_vector(vel)


            # If not measureing bnu set nu equal bnu, forcing correct dynamics through input
            if not self.measure_b_mu:
                u.mu = xi_hat.b[6:9, 0:1]
                # print(f"nu: {u.mu.T}")

            lift = continuous_lift(xi_hat, u)
            A0t = self.stateMatrixA_CT(u)
            Bt = self.inputMatrixBt_CT(u)

            # Propagation
            M = Bt @ R @ Bt.T

            Phi_DT = expm(A0t*dt)
            self.X_hat = self.X_hat * SymGroup.exp(lift * dt)
            self.Sigma = Phi_DT @ self.Sigma @ Phi_DT.T + M * dt

        # Update when measurement available and if allowed
        if not self.propagation_only:
            if not np.isnan(y[0, 0:1]):

                if self.measure_b_mu:
                    Ct = np.zeros((6, 18))
                    Ct[3:6, 9:12] = self.X_hat.B.R().inv().as_matrix() @ SO3.wedge(self.X_hat.B.w().as_vector())
                    Ct[3:6, 15:18] = -self.X_hat.B.R().inv().as_matrix()
                else:
                    Ct = np.zeros((3, 18))

                if self.equivariant_output:
                    Ct[0:3, 0:3] = 0.5 * SO3.wedge(y + self.X_hat.B.w().as_vector())
                    # Ct[0:3, 0:3] = SO3.wedge(y)
                    Ct[0:3, 6:9] = -np.eye(3)
                    if self.measure_b_mu:
                        delta = np.vstack((self.X_hat.B.w().as_vector() - y, np.zeros((3, 1)) - xi_hat.b[6:9, 0:1]))
                    else:
                        delta = self.X_hat.B.w().as_vector() - y  # rho_hat{X^{-1}}(0)-y
                else:
                    Ct[0:3, 0:3] = -SO3.wedge(xi_hat.T.w().as_vector())
                    Ct[0:3, 6:9] = np.eye(3)
                    if self.measure_b_mu:
                        delta = np.vstack((y, np.zeros((3, 1)))) - measurePosAndBias(xi_hat)
                    else:
                        delta = y - measurePos(xi_hat)

                S = Ct @ self.Sigma @ Ct.T + Q
                Sinv = np.linalg.inv(S)
                nis = delta.T @ Sinv @ delta
                K = self.Sigma @ Ct.T @ Sinv
                Delta = self.InnovationLift @ K @ delta
                self.X_hat = SymGroup.exp(Delta) * self.X_hat
                self.Sigma = (np.eye(18) - K @ Ct) @ self.Sigma
                if self.curvature_correction:
                    Gamma = 0.5 * grp_adj(K @ delta)
                    exp_Gamma = expm(Gamma)
                    self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
                return nis

    def stateMatrixA_CT(self, u : InputSpace) -> np.ndarray:

        u_0 = velocityAction(self.X_hat.inv(), u)
        # stf = lambda eps : self.Dphi0 @ continuous_lift(local_coords_inv(eps), u_0)
        # A0t = numericalDifferential(stf, np.zeros((18, 1)))

        A0t = np.zeros((18, 18))
        A0t[0:9, 0:9] = np.hstack((blockDiag(np.vstack((np.zeros((3, 3)), SO3.skew(np.vstack((0.0, 0.0, -9.81))))), np.eye(3)), np.zeros((9, 3))))
        A0t[9:18, 9:18] = SE23.adjoint(u_0.as_W_vec() + SE23.vee(G))
        A0t[0:9, 9:18] = np.eye(9)

        # If exponential_coordinates are used then multiply by -1
        if exponential_coords == True:
            A0t[0:9, 9:18] = -A0t[0:9, 9:18]

        return A0t

    def inputMatrixBt_CT(self, u : InputSpace) -> np.ndarray:
        # l0 = continuous_lift(xi_0, velocityAction(self.X_hat.inv(), u))
        # stf = lambda n: self.Dphi0 @ (continuous_lift(xi_0, velocityAction(self.X_hat.inv(), input_from_vector(u.as_vector() + n))) - l0)
        # Bt = numericalDifferential(stf, np.zeros((18, 1)))
        Bt = np.zeros((18, 18))
        Bt[0:9, 0:9] = self.X_hat.B.Adjoint()
        Bt[9:18, 9:18] = self.X_hat.B.Adjoint()
        return Bt

    # Ct = [-SO3.wedge(xi_hat.T.w().as_vector()), 0, I, 0, 0, 0]
    #      [0,                                    0, 0, ?, 0, ?]
    def outputMatrixC(self, y: np.ndarray) -> np.ndarray:
        if self.measure_b_mu:
            opf = lambda eps : measurePosAndBias(stateAction(self.X_hat, local_coords_inv(eps)))
        else:
            opf = lambda eps : measurePos(stateAction(self.X_hat, local_coords_inv(eps)))
        C0 = numericalDifferential(opf, np.zeros((18, 1)))
        # if self.measure_b_mu:
        #     C0 = np.zeros((6, 18))
        #     C0[0:3, 0:3] = 0.5*SO3.wedge(y + self.X_hat.B.w().as_vector())
        #     C0[0:3, 6:9] = -np.eye(3)
        #     C0[3:6, 9:12] = -self.X_hat.B.R().inv().as_matrix()
        #     C0[3:6, 15:18] = self.X_hat.B.R().inv().as_matrix() @ SO3.wedge(self.X_hat.B.w().as_vector())
        # else:
        #     C0 = np.zeros((3, 18))
        #     C0[0:3, 0:3] = 0.5 * SO3.wedge(y + self.X_hat.B.w().as_vector())
        #     C0[0:3, 6:9] = -np.eye(3)
        return C0

    def computeNEES(self, xi):
        xi_state = stateFromData(xi)
        e = stateAction(self.X_hat.inv(), xi_state)        
        eps = local_coords(e)
        ness = eps.T @ np.linalg.inv(self.Sigma) @ eps
        nav_ness = eps[0:9, :].T @ np.linalg.inv(self.Sigma[0:9, 0:9]) @ eps[0:9, :]
        bias_ness = eps[9:self.dof, :].T @ np.linalg.inv(self.Sigma[9:self.dof, 9:self.dof]) @ eps[9:self.dof, :]
        return float(ness), float(nav_ness), float(bias_ness)


class TestGroup(unittest.TestCase):
    test_reps = 100

    def setUp(self) -> None:
        np.random.seed(0)
        return super().setUp()

    def assertStateEqual(self, S1: State, S2: State):
        sr1 = S1.T.as_matrix()
        sr2 = S2.T.as_matrix()
        self.assertMatricesEqual(sr1, sr2)
        self.assertMatricesEqual(S1.b, S2.b)

    def assertMatricesEqual(self, M1: np.ndarray, M2: np.ndarray):
        assert (M1.shape == M2.shape)
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                self.assertAlmostEqual(M1[i, j], M2[i, j])

    def assertSymGroupEqual(self, X1: SymGroup, X2: SymGroup):
        self.assertMatricesEqual(X1.B.as_matrix(), X2.B.as_matrix())
        self.assertMatricesEqual(X1.beta, X2.beta)

    def test_exp_log(self):
        for t in range(TestGroup.test_reps):
            x = np.random.randn(18, 1)
            #
            xw = np.zeros((10, 10))
            xw[0:9, 0:9] = SE23.adjoint(x[0:9,:])
            xw[0:9, 9:10] = x[9:18, :]
            Xexpm = expm(xw)
            AdB = Xexpm[0:9, 0:9]
            beta = Xexpm[0:9, 9:10]
            #
            X = SymGroup.exp(x)
            #
            self.assertMatricesEqual(X.B.Adjoint(), AdB)
            self.assertMatricesEqual(SE23.vee(X.beta), beta)

    def test_associative(self):
        for t in range(TestGroup.test_reps):
            X1 = SymGroup.random()
            X2 = SymGroup.random()
            X3 = SymGroup.random()
            Z1 = (X1 * X2) * X3
            Z2 = X1 * (X2 * X3)
            self.assertSymGroupEqual(Z1, Z2)

    def test_inverse_identity(self):

        for t in range(TestGroup.test_reps):
            X = SymGroup.random()
            XInv = X.inv()
            I = SymGroup.identity()
            I1 = X * XInv
            I2 = XInv * X
            self.assertSymGroupEqual(I, I1)
            self.assertSymGroupEqual(I, I2)
            self.assertSymGroupEqual(I1, I2)
            X1 = X * I
            X2 = I * X
            self.assertSymGroupEqual(X, X1)
            self.assertSymGroupEqual(X, X2)
            self.assertSymGroupEqual(X1, X2)
            XInv1 = XInv * I
            XInv2 = I * XInv
            self.assertSymGroupEqual(XInv, XInv1)
            self.assertSymGroupEqual(XInv, XInv2)
            self.assertSymGroupEqual(XInv1, XInv2)

    def test_f_10(self):
        for t in range(TestGroup.test_reps):
            X1 = SymGroup.random().B.as_matrix()
            X2 = SymGroup.random().B.as_matrix()
            f = X1 @ f_10(X2) + f_10(X1)
            self.assertMatricesEqual(f_10(X1 @ X2), f)

    def test_velocity_action(self):
        for t in range(TestGroup.test_reps):
            X1 = SymGroup.random()
            X2 = SymGroup.random()
            U = InputSpace.random()
            U0 = velocityAction(SymGroup.identity(), U)
            self.assertMatricesEqual(U0.as_vector(), U.as_vector())
            U1 = velocityAction(X2, velocityAction(X1, U))
            U2 = velocityAction(X1 * X2, U)
            self.assertMatricesEqual(U1.as_vector(), U2.as_vector())

    def test_state_action(self):
        for t in range(TestGroup.test_reps):
            X1 = SymGroup.random()
            X2 = SymGroup.random()
            xi = State.random()
            xi0 = stateAction(SymGroup.identity(), xi)
            self.assertMatricesEqual(xi0.vec(), xi.vec())
            xi1 = stateAction(X2, stateAction(X1, xi))
            xi2 = stateAction(X1 * X2, xi)
            self.assertMatricesEqual(xi1.vec(), xi2.vec())

    def continuousTimeDinamics(self, xi : State, U : InputSpace) -> np.ndarray:
        dot_T = SE23.vee(xi.T.as_matrix() @ (U.as_W_mat() - SE23.wedge(xi.b)) + G + f_10(xi.T.as_matrix()))
        dot_b = U.tau
        return np.vstack((dot_T, dot_b))

    def test_lift(self):
        for t in range(TestGroup.test_reps):
            xi = State()
            U = InputSpace.random()
            f1 = self.continuousTimeDinamics(xi, U)
            f2 = lambda dt: stateAction(SymGroup.exp(continuous_lift(xi, U) * dt), xi).vec()
            Df2 = numericalDifferential(f2, 0.0)
            self.assertMatricesEqual(f1, Df2)

    def test_lift_equivariance(self):
        for t in range(TestGroup.test_reps):
            xi = State()
            U = InputSpace.random()
            X = SymGroup.random()
            f1 = continuous_lift(xi, U)
            f2 = lambda dt: (X * SymGroup.exp(continuous_lift(stateAction(X, xi), velocityAction(X, U)) * dt) * X.inv()).vec()
            Df2 = numericalDifferential(f2, 0.0)
            self.assertMatricesEqual(f1, Df2)

    def test_coords(self):
        for _ in range(self.test_reps):
            eps = 0.1*np.random.randn(18, 1)
            xi = local_coords_inv(eps)
            eps1 = local_coords(xi)
            self.assertMatricesEqual(eps, eps1)

    def test_adj(self):
        for _ in range(self.test_reps):
            x = np.random.randn(18, 1)
            #
            x_wedge = np.zeros((10, 10))
            x_wedge[0:9, 0:9] = SE23.adjoint(x[0:9,:])
            x_wedge[0:9, 9:10] = x[9:18, :]
            X = expm(x_wedge)
            AdB = X[0:9,0:9]
            beta = X[0:9,9:10]
            #
            Ad = np.zeros((18, 18))
            Ad[0:9, 0:9] = AdB
            Ad[9:18, 9:18] = AdB
            Ad[9:18, 0:9] = SE23.adjoint(beta) @ AdB
            ad = grp_adj(x)
            self.assertMatricesEqual(Ad, expm(ad))

    def test_matrix_bv(self):
        for _ in range(self.test_reps):
            X = SymGroup.random()
            opf = lambda eps: measurePosAndBias(stateAction(X, local_coords_inv(eps)))
            C0 = numericalDifferential(opf, np.zeros((18, 1)))
            Cn = C0[3:6, :]

            Ca = np.zeros((3, 18))
            Ca[:, 0:3] = X.B.R().as_matrix().T @ (xi_0.b[6:9, :] - SO3.wedge(X.B.w().as_vector()) @ xi_0.b[0:3, :])
            Ca[:, 6:9] = X.B.R().as_matrix().T @ SO3.wedge(xi_0.b[0:3, :])
            Ca[:, 9:12] = -X.B.R().as_matrix().T
            Ca[:, 15:18] = X.B.R().as_matrix().T @ SO3.wedge(X.B.w().as_vector())

            pdb.set_trace()

if __name__ == "__main__":

    test = TestGroup()

    # print("*******************************")
    # print("Running symmetry test!")
    # print("*******************************")
    # print("*******************************")
    # print("Associativity test...")
    # print("*******************************")
    # test.test_associative()
    # print("*******************************")
    # print("Inverse test...")
    # print("*******************************")
    # test.test_inverse_identity()
    # print("*******************************")
    # print("Test f_10 ...")
    # print("*******************************")
    # test.test_f_10()
    # print("*******************************")
    # print("Velocity action test ...")
    # print("*******************************")
    # test.test_velocity_action()
    # print("*******************************")
    # print("State action test ...")
    # print("*******************************")
    # test.test_state_action()
    #
    # print("*******************************")
    # print("Running equivariance lift test!")
    # print("*******************************")
    # print("*******************************")
    # print("Lift test ...")
    # print("*******************************")
    # test.test_lift()
    # print("*******************************")
    # print("Lift equivariance test ...")
    # print("*******************************")
    # test.test_lift_equivariance()
    # print("*******************************")
    # print("Lift is equivariant!")
    # print("*******************************")
    #
    # print("*******************************")
    # print("Running coordinates test!")
    # print("*******************************")
    # test.test_coords()
    # print("*******************************")
    # print("Coords are fine!")
    # print("*******************************")

    # print("*******************************")
    # print("Running test for C for bv!")
    # print("*******************************")
    # test.test_matrix_bv()

    # print("*******************************")
    # print("Running test for Adjoints!")
    # print("*******************************")
    test.test_exp_log()
    test.test_adj()
