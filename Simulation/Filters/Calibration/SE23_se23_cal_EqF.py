# Add main directory to path
import pdb
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from Symmetries.Calibration.SE23_se23.Symmetry import *
from scipy.linalg import expm
import unittest
from pylie import SE23


def continuous_lift(xi : State, U : InputSpace) -> np.ndarray:
    L = np.zeros((21, 1))
    L[0:9, 0:1] = (U.as_W_vec() - xi.b) + SE23.vee(xi.T.inv().as_matrix() @ (G + f_10(xi.T.as_matrix())))
    L[9:18, 0:1] = SE23.adjoint(xi.b) @ L[0:9, 0:1] - U.tau
    L[18:21, 0:1] = -(U.eta + ((U.w - SO3.wedge(xi.b[0:3, 0:1])) @ xi.t))
    return L


class SE23_se23_cal_EqF:
    def __init__(self, initial_nav_noise=1.0, initial_bias_noise=0.01, propagationonly=False, curvature_correction=False, measure_b_mu=False):
        self.X_hat = SymGroup()

        sigma_vec = np.concatenate((np.ones((1, 9)) * initial_nav_noise ** 2, np.ones((1, 9)) * initial_bias_noise ** 2, np.ones((1, 3)) * (initial_nav_noise / 5) ** 2), axis=1)

        self.Sigma = np.eye(sigma_vec.shape[1]) * sigma_vec
        self.Dphi0 = stateActionDiff(xi_0)              # (D\theta) * (D\phi_{xi_0}(E) in E = Id)
        self.InnovationLift = np.linalg.pinv(self.Dphi0)
        self.propagation_only = propagationonly
        self.measure_b_mu = measure_b_mu
        self.curvature_correction = curvature_correction
        self.dof = 21
        if self.measure_b_mu:
            print("Measuring b_mu")
        if self.curvature_correction:
            print("Curvature correction enabled")
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
        t = xi_hat.t
        return R, p, v, bw, ba, bmu, t

    def update(self, vel : np.ndarray, omega_noise : float, acc_noise : float, tau_noise : float, virtual_noise : float, y : np.ndarray, meas_noise : float, dt : float):

        # Settings
        noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 3)) * virtual_noise ** 2, np.ones((1, 9)) * (tau_noise) ** 2, np.ones((1, 3)) * virtual_noise ** 2), axis=1)
        R = np.eye(noise_vec.shape[1]) * noise_vec
        Q = np.eye(3) * (meas_noise**2)
        if self.measure_b_mu:
            Q = blockDiag(Q, np.eye(3) * 1e-9)

        # Input (w, a, mu, tau)
        u = input_from_vector(vel)

        # Filter matrices
        xi_hat = self.stateEstimate()       #phi_{xi_0}(\hat{X})
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
                xi_hat = self.stateEstimate()
                if self.measure_b_mu:
                    Ct1 = self.outputMatrixC()
                    Ct2 = np.hstack((0.5*(SO3.wedge(y + (self.X_hat.B.w().as_vector() - SO3.vee(self.X_hat.gamma)))),
                                     np.zeros((3, 3)),
                                     -np.eye(3),
                                     np.zeros((3, 9)),
                                     np.eye(3))) # C* trick
                    Ct = np.vstack((Ct1, Ct2))
                    delta = np.vstack((-measureBias(xi_hat), outputAction(self.X_hat.inv(), np.zeros((3, 1))) - y)) # trick
                else:
                    Ct = np.hstack((0.5*(SO3.wedge(y + (self.X_hat.B.w().as_vector() - SO3.vee(self.X_hat.gamma)))),
                                    np.zeros((3, 3)),
                                    -np.eye(3),
                                    np.zeros((3, 9)),
                                    np.eye(3))) # C* trick
                    delta = outputAction(self.X_hat.inv(), np.zeros((3, 1))) - y # trick
                S = Ct @ self.Sigma @ Ct.T + Q
                K = self.Sigma @ Ct.T @ np.linalg.inv(S)
                Delta = self.InnovationLift @ K @ delta
                self.X_hat = SymGroup.exp(Delta) * self.X_hat
                self.Sigma = (np.eye(self.dof) - K @ Ct) @ self.Sigma
                if self.curvature_correction:
                    Gamma = 0.5 * grp_adj(K @ delta)
                    exp_Gamma = expm(Gamma)
                    self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T

    def stateMatrixA_CT(self, u : InputSpace) -> np.ndarray:
        u_0 = velocityAction(self.X_hat.inv(), u)
        # stf = lambda eps : self.Dphi0 @ continuous_lift(local_coords_inv(eps), u_0)
        # A0t = numericalDifferential(stf, np.zeros((21, 1)))

        A0t = np.zeros((21, 21))
        A0t[0:9, 0:9] = np.hstack((blockDiag(np.vstack((np.zeros((3, 3)), SO3.skew(np.vstack((0.0, 0.0, -9.81))))), np.eye(3)), np.zeros((9, 3))))
        A0t[9:18, 9:18] = SE23.adjoint(u_0.as_W_vec() + SE23.vee(G))
        A0t[0:9, 9:18] = np.eye(9)
        A0t[18:21, 18:21] = u_0.w

        # If exponential_coordinates are used then multiply by -1
        if exponential_coords == True:
            A0t[0:9, 9:18] = -A0t[0:9, 9:18]

        return A0t

    def inputMatrixBt_CT(self, u : InputSpace) -> np.ndarray:
        # l0 = continuous_lift(xi_0, velocityAction(self.X_hat.inv(), u))
        # stf = lambda n: self.Dphi0 @ (continuous_lift(xi_0, velocityAction(self.X_hat.inv(), input_from_vector(u.as_vector() + n))) - l0)
        # Bt = numericalDifferential(stf, np.zeros((21, 1)))

        Bt = np.zeros((21, 21))
        Bt[0:9, 0:9] = self.X_hat.B.Adjoint()
        Bt[9:18, 9:18] = -self.X_hat.B.Adjoint()
        Bt[18:21, 18:21] = -self.X_hat.B.R().as_matrix()

        return Bt

    # (1) = -(SO3.wedge(self.X_hat.B.w().as_vector()) - self.X_hat.gamma)
    # Ct = [(1), 0, I, 0, 0, 0, -I]
    #      [0,   0, 0, ?, 0, ?,  0]
    def outputMatrixC(self) -> np.ndarray:
        if self.measure_b_mu:
            opf = lambda eps : measureBias(stateAction(self.X_hat, local_coords_inv(eps)))
        else:
            opf = lambda eps : measurePos(stateAction(self.X_hat, local_coords_inv(eps)))
        C0 = numericalDifferential(opf, np.zeros((21, 1)))
        return C0

    def computeError(self, xi) -> float:
        xi_state = stateFromData(xi)
        e = stateAction(self.X_hat.inv(), xi_state)        
        eps = local_coords(e)
        err = eps.T @ np.linalg.inv(self.Sigma) @ eps
        return float(err)


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
        self.assertMatricesEqual(S1.t, S2.t)

    def assertMatricesEqual(self, M1: np.ndarray, M2: np.ndarray):
        assert (M1.shape == M2.shape)
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                self.assertAlmostEqual(M1[i, j], M2[i, j])

    def assertSymGroupEqual(self, X1: SymGroup, X2: SymGroup):
        self.assertMatricesEqual(X1.B.as_matrix(), X2.B.as_matrix())
        self.assertMatricesEqual(X1.beta, X2.beta)
        self.assertMatricesEqual(X1.gamma, X2.gamma)

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
        dot_t = U.eta
        return np.vstack((dot_T, dot_b, dot_t))

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
            eps = 0.1*np.random.randn(21, 1)
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
    print("Test f_10 ...")
    print("*******************************")
    test.test_f_10()
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
    print("Running coordinates test!")
    print("*******************************")
    test.test_coords()
    print("*******************************")
    print("Coords are fine!")
    print("*******************************")
