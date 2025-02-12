# Add main directory to path
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from Symmetries.Calibrated.SE3_se3_R3.Symmetry import *
from scipy.linalg import expm
import unittest
from pylie import SE3


def continuous_lift(xi : State, U : InputSpace) -> np.ndarray:
    L = np.zeros((15, 1))
    L[0:6, 0:1] = (U.as_wa_vec() - xi.b) + SE3.vee(xi.T.inv().as_matrix() @ G)
    L[6:12, 0:1] = SE3.adjoint(xi.b) @ L[0:6, 0:1] - U.tau
    L[12:15, 0:1] = xi.T.R().as_matrix() @ U.mu + xi.T.x().as_vector()
    return L


class SE3_se3_R3_EqF:
    def __init__(self, initial_att_noise=1.0, initial_vel_noise=1.0, initial_pos_noise=1.0, initial_bias_noise=0.01, propagationonly=False, curvature_correction=False):
        self.X_hat = SymGroup()
        sigma_vec = np.concatenate((
            np.ones((1, 3)) * initial_att_noise**2,
            np.ones((1, 3)) * initial_vel_noise**2,
            np.ones((1, 6)) * initial_bias_noise**2,
            np.ones((1, 3)) * initial_pos_noise**2), axis=1)
        self.Sigma = np.eye(sigma_vec.shape[1]) * sigma_vec
        self.Dphi0 = stateActionDiff(xi_0)              # (D\theta) * (D\phi_{xi_0}(E) in E = Id)
        self.InnovationLift = np.linalg.pinv(self.Dphi0)
        self.propagation_only = propagationonly
        self.curvature_correction = curvature_correction
        self.dof = 15
        self.t = None
        self.u = None
        if self.curvature_correction:
            print("Curvature correction enabled")
        print(f"Exponential coordinates = {exponential_coords}")
        print(f"Fake exponential = {fake_exponential}")
    
    def stateEstimate(self):
        return stateAction(self.X_hat, xi_0)

    def getEstimate(self):
        xi_hat = self.stateEstimate()
        R = xi_hat.T.R().as_matrix()
        p = xi_hat.p
        v = xi_hat.T.x().as_vector()
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
        noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2,
                                    np.ones((1, 6)) * (tau_noise) ** 2, np.ones((1, 3)) * virtual_noise ** 2), axis=1)
        R = np.eye(noise_vec.shape[1]) * noise_vec

        # Filter matrices
        xi_hat = self.stateEstimate()  # phi_{xi_0}(\hat{X})
        lift = continuous_lift(xi_hat, self.u)
        A0t = self.stateMatrixA_CT(self.u)
        Bt = self.inputMatrixBt_CT(self.u)

        # Propagation
        # M = (np.eye(15) + Bt*dt) @ (R / dt) @ (np.eye(15) + Bt*dt).T
        M = Bt @ R @ Bt.T

        Phi_DT = expm(A0t * dt)
        self.X_hat = self.X_hat * SymGroup.exp(lift * dt)
        self.Sigma = Phi_DT @ self.Sigma @ Phi_DT.T + M * dt

        self.t = t
        self.u = input_from_vector(vel)

        return False

    # def update(self, y: np.ndarray, meas_noise: float):
    #     Q = np.eye(3) * (meas_noise ** 2)
    #
    #     if not self.propagation_only:
    #         if not np.isnan(y[0,0:1]):
    #             # if exponential_coords:
    #             #     Ct = self.outputMatrixC()
    #             # else:
    #             #     Ct = self.outputMatrixCStar(y)
    #             Ct = np.hstack((np.zeros((3, 12)), np.eye(3)))
    #             delta = outputAction(self.X_hat.inv(), y)
    #             # rhoXInv = lambda u : outputAction(self.X_hat.inv(), y + u[0:3, :]) # Given the action the diff is the Id
    #             # drhoXInv = numericalDifferential(rhoXInv, np.zeros((3,1)))
    #             # Q  = drhoXInv @ Q @ drhoXInv.T
    #             S = Ct @ self.Sigma @ Ct.T + Q
    #             Sinv = np.linalg.inv(S)
    #             nis = delta.T @ Sinv @ delta
    #             K = self.Sigma @ Ct.T @ Sinv
    #             Delta = self.InnovationLift @ K @ delta
    #             self.X_hat = SymGroup.exp(Delta) * self.X_hat
    #             self.Sigma = (np.eye(15) - K @ Ct) @ self.Sigma
    #             if self.curvature_correction:
    #                 Gamma = 0.5 * grp_adj(K @ delta)
    #                 exp_Gamma = expm(Gamma)
    #                 self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
    #             return nis

    def update(self, vel : np.ndarray, omega_noise : float, acc_noise : float, tau_noise : float, virtual_noise : float, y : np.ndarray, meas_noise : float, dt : float, propagate : bool = True):

        # Settings
        Q = np.eye(3) * (meas_noise ** 2)

        if propagate:
            noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 6)) * (tau_noise) ** 2, np.ones((1, 3)) * virtual_noise ** 2), axis=1)
            R = np.eye(noise_vec.shape[1]) * noise_vec

            # Filter matrices
            u = input_from_vector(vel)
            xi_hat = self.stateEstimate()       #phi_{xi_0}(\hat{X})
            lift = continuous_lift(xi_hat, u)
            A0t = self.stateMatrixA_CT(u)
            Bt = self.inputMatrixBt_CT(u)

            # Propagation
            # M = (np.eye(15) + Bt*dt) @ (R / dt) @ (np.eye(15) + Bt*dt).T
            M = Bt @ R @ Bt.T

            Phi_DT = expm(A0t*dt)
            self.X_hat = self.X_hat * SymGroup.exp(lift * dt)
            self.Sigma = Phi_DT @ self.Sigma @ Phi_DT.T + M * dt

        # Update when measurement available and if allowed
        if not self.propagation_only:
            if not np.isnan(y[0,0:1]):
                # if exponential_coords:
                #     Ct = self.outputMatrixC()
                # else:
                #     Ct = self.outputMatrixCStar(y)
                Ct = np.hstack((np.zeros((3, 12)), np.eye(3)))
                delta = outputAction(self.X_hat.inv(), y)
                # rhoXInv = lambda u : outputAction(self.X_hat.inv(), y + u[0:3, :]) # Given the action the diff is the Id
                # drhoXInv = numericalDifferential(rhoXInv, np.zeros((3,1)))
                # Q  = drhoXInv @ Q @ drhoXInv.T
                S = Ct @ self.Sigma @ Ct.T + Q
                Sinv = np.linalg.inv(S)
                nis = delta.T @ Sinv @ delta
                K = self.Sigma @ Ct.T @ Sinv
                Delta = self.InnovationLift @ K @ delta
                self.X_hat = SymGroup.exp(Delta) * self.X_hat
                self.Sigma = (np.eye(15) - K @ Ct) @ self.Sigma
                if self.curvature_correction:
                    Gamma = 0.5 * grp_adj(K @ delta)
                    exp_Gamma = expm(Gamma)
                    self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
                return nis

    def stateMatrixA_CT(self, u : InputSpace) -> np.ndarray:

        u_0 = velocityAction(self.X_hat.inv(), u)
        # stf = lambda eps : self.Dphi0 @ continuous_lift(local_coords_inv(eps), u_0)
        # A0t = numericalDifferential(stf, np.zeros((15,1)))

        A0t = np.zeros((15, 15))
        Psi = np.zeros((6, 6))
        Psi[3:6, 0:3] = SO3.skew(np.vstack((0.0, 0.0, -9.81)))
        Gamma = np.hstack((-SO3.wedge(u_0.mu), np.eye(3)))

        A0t[0:6, 0:6] = Psi
        A0t[12:15, 0:6] = Gamma
        A0t[0:6, 6:12] = np.eye(6)
        A0t[6:12, 6:12] = SE3.adjoint(u_0.as_wa_vec() + SE3.vee(G))

        # If exponential_coordinates are used then multiply by -1
        if exponential_coords == True:
            A0t[0:6, 6:12] = -A0t[0:6, 6:12]

        return A0t

    def inputMatrixBt_CT(self, u : InputSpace) -> np.ndarray:     
        # l0 = continuous_lift(xi_0,velocityAction(self.X_hat.inv(),u))
        # stf = lambda n: self.Dphi0 @ (continuous_lift(xi_0, velocityAction(self.X_hat.inv(), input_from_vector(u.as_vector() + n)))-l0)
        # Bt = numericalDifferential(stf, np.zeros((15,1)))
        Bt = np.zeros((15, 15))
        Bt[0:6, 0:6] = self.X_hat.B.Adjoint()
        Bt[6:12, 6:12] = self.X_hat.B.Adjoint()
        Bt[12:15, 12:15] = self.X_hat.B.R().as_matrix()
        return Bt

    def outputMatrixC(self) -> np.ndarray:
        opf = lambda eps : measurePos(stateAction(self.X_hat, local_coords_inv(eps)))
        C0 = numericalDifferential(opf, np.zeros((15, 1)))
        return C0

    def outputMatrixCStar(self, y : np.ndarray) -> np.ndarray:
        opf_1 = lambda U: outputAction(SymGroup.exp(U), measurePos(xi_0))
        opf_2 = lambda U: outputAction(SymGroup.exp(U), outputAction(self.X_hat.inv(), y))
        CStar_1 = numericalDifferential(opf_1, np.zeros((15,1)))
        CStar_2 = numericalDifferential(opf_2, np.zeros((15, 1)))
        return 0.5 * (CStar_1 + CStar_2)

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
        self.assertMatricesEqual(S1.p, S2.p)
        self.assertMatricesEqual(S1.b, S2.b)

    def assertMatricesEqual(self, M1: np.ndarray, M2: np.ndarray):
        assert (M1.shape == M2.shape)
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                self.assertAlmostEqual(M1[i, j], M2[i, j])

    def assertSymGroupEqual(self, X1: SymGroup, X2: SymGroup):
        self.assertMatricesEqual(X1.B.as_matrix(), X2.B.as_matrix())
        self.assertMatricesEqual(X1.beta, X2.beta)
        self.assertMatricesEqual(X1.z, X2.z)

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

    def test_velocity_action(self):
        for t in range(TestGroup.test_reps):
            X1 = SymGroup.random()
            X2 = SymGroup.random()
            U = InputSpace.random()
            U0 = velocityAction(SymGroup.identity(), U)
            self.assertMatricesEqual(U0.as_wa_mat(), U.as_wa_mat())
            self.assertMatricesEqual(U0.tau, U.tau)
            U1 = velocityAction(X2, velocityAction(X1, U))
            U2 = velocityAction(X1 * X2, U)
            self.assertMatricesEqual(U1.as_wa_mat(), U2.as_wa_mat())
            self.assertMatricesEqual(U1.tau, U2.tau)

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
        dot_V = SE3.vee(xi.T.as_matrix() @ (U.as_wa_mat() - SE3.wedge(xi.b)) + G)
        dot_p = xi.T.R().as_matrix() @ U.mu + xi.T.x().as_vector()
        dot_b = U.tau
        return np.vstack((dot_V, dot_b, dot_p))

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
            f2 = lambda dt: (X * SymGroup.exp(continuous_lift(stateAction(X,xi), velocityAction(X,U)) * dt) * X.inv()).vec()
            Df2 = numericalDifferential(f2, 0.0)
            self.assertMatricesEqual(f1, Df2)

    def test_coords(self):
        for _ in range(self.test_reps):
            eps = 0.1*np.random.randn(15, 1)
            xi = local_coords_inv(eps)
            eps1 = local_coords(xi)
            self.assertMatricesEqual(eps, eps1)


if __name__ == "__main__":

    test = TestGroup()

    print("*******************************")
    print("Running symmetry test!")
    print("*******************************")
    print("*******************************")
    print("Associativity test...")
    print("*******************************")
    test.test_associative()
    print("*******************************")
    print("Inverse test...")
    print("*******************************")
    test.test_inverse_identity()
    print("*******************************")
    print("Velocity action test ...")
    print("*******************************")
    test.test_velocity_action()
    print("*******************************")
    print("State action test ...")
    print("*******************************")
    test.test_state_action()

    print("*******************************")
    print("Running equivariance lift test!")
    print("*******************************")
    print("*******************************")
    print("Lift test ...")
    print("*******************************")
    test.test_lift()
    print("*******************************")
    print("Lift equivariance test ...")
    print("*******************************")
    test.test_lift_equivariance()
    print("*******************************")
    print("Lift is equivariant!")
    print("*******************************")

    print("*******************************")
    print("Running coordinates lift test!")
    print("*******************************")
    test.test_coords()
    print("*******************************")
    print("Coords are fine!")
    print("*******************************")
