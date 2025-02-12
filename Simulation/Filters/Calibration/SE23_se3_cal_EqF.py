# Add main directory to path
import pdb
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from Symmetries.Calibration.SE23_se3.Symmetry import *
from scipy.linalg import expm
import unittest
from pylie import SE3


def continuous_lift(xi : State, U : InputSpace) -> np.ndarray:
    L = np.zeros((18, 1))
    L[0:9, 0:1] = SE23.vee(x_o(SE3.wedge(U.as_wa_vec() - xi.b)) + D + xi.T.inv().as_matrix() @ (G - D))
    L[9:15, 0:1] = SE3.adjoint(xi.b) @ L[0:6, 0:1] - U.tau
    L[15:18, 0:1] = -(U.eta + ((U.w - SO3.wedge(xi.b[0:3, 0:1])) @ xi.t))
    return L


class SE23_se3_cal_EqF:
    def __init__(self, initial_nav_noise=1.0, initial_bias_noise=0.01, propagationonly=False, curvature_correction=False):
        self.X_hat = SymGroup()

        sigma_vec = np.concatenate((np.ones((1, 9)) * initial_nav_noise ** 2, np.ones((1, 6)) * initial_bias_noise ** 2, np.ones((1, 3)) * (initial_nav_noise / 5) ** 2), axis=1)
        self.Sigma = np.eye(sigma_vec.shape[1]) * sigma_vec
        self.Dphi0 = stateActionDiff(xi_0)  # (D\theta) * (D\phi_{xi_0}(E) in E = Id)
        self.InnovationLift = np.linalg.pinv(self.Dphi0)
        self.propagation_only = propagationonly
        self.curvature_correction = curvature_correction
        self.dof = 18
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
        t = xi_hat.t
        return R, p, v, bw, ba, np.zeros((3, 1)), t

    def update(self, vel : np.ndarray, omega_noise : float, acc_noise : float, tau_noise : float, virtual_noise : float, y : np.ndarray, meas_noise : float, dt : float):

        # Settings
        noise_vec = np.concatenate((np.ones((1, 3)) * omega_noise ** 2, np.ones((1, 3)) * acc_noise ** 2, np.ones((1, 6)) * (tau_noise) ** 2, np.ones((1, 3)) * virtual_noise ** 2), axis=1)
        R = np.eye(noise_vec.shape[1]) * noise_vec
        Q = np.eye(3) * (meas_noise**2)

        # Filter matrices
        u = input_from_vector(vel)
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
            if not np.isnan(y[0,0:1]):
                # Ct = self.outputMatrixC()
                Ct = np.hstack((-(SO3.wedge(self.X_hat.B.w().as_vector()) - self.X_hat.gamma),
                                np.zeros((3, 3)),
                                np.eye(3),
                                np.zeros((3, 6)),
                                np.eye(3)))
                delta = y - measurePos(xi_hat)
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

        xi_hat = self.stateEstimate()

        # # (D\theta) * (D\phi_{\hat{X}}) * (D\phi_{\hat{xi}}(E)) in E = Id)
        # coordsAction = lambda U: local_coords(stateAction(self.X_hat.inv(), stateAction(SymGroup.exp(U), xi_hat)))
        # Dphi = numericalDifferential(coordsAction, np.zeros((18, 1)))
        #
        # stf = lambda eps : Dphi @ continuous_lift(stateAction(self.X_hat, local_coords_inv(eps)), u)
        # A0t = numericalDifferential(stf, np.zeros((18, 1)))

        A0t = np.zeros((18, 18))
        A0t[0:9, 0:9] = np.hstack((blockDiag(np.vstack((np.zeros((3, 3)), SO3.skew(g))), np.eye(3)), np.zeros((9, 3))))
        A0t[9:15, 9:15] = SE3.adjoint(xi_hat.A().Adjoint() @ (u.as_wa_vec() - xi_hat.b) + SE3.vee(G[0:4, 0:4]))
        A0t[0:6, 9:15] = np.eye(6)
        A0t[6:9, 9:12] = SO3.wedge(xi_hat.T.w().as_vector())
        A0t[15:18, 15:18] = SO3.wedge(xi_hat.T.R().as_matrix() @ SO3.vee(u.w) + SE3.vee(self.X_hat.beta)[0:3, :])

        # If exponential_coordinates are used then multiply by -1
        if exponential_coords == True:
            A0t[0:6, 9:15] = -A0t[0:6, 9:15]
            A0t[6:9, 9:12] = -A0t[6:9, 9:12]

        return A0t

    def inputMatrixBt_CT(self, u : InputSpace) -> np.ndarray:

        # xi_hat = self.stateEstimate()
        # stf = lambda n: self.Dphi0 @ Adj_action(self.X_hat, continuous_lift(xi_hat, input_from_vector(u.as_vector() + n)))
        # Bt = numericalDifferential(stf, np.zeros((15, 1)))

        Bt = np.zeros((18, 15))
        tmp = self.X_hat.B.Adjoint()
        Bt[0:9, 0:6] = tmp[:, 0:6]
        Bt[9:15, 6:12] = -tmp[0:6, 0:6]
        Bt[15:18, 12:15] = -self.X_hat.B.R().as_matrix()

        return Bt

    def outputMatrixC(self) -> np.ndarray:
        opf = lambda eps : measurePos(stateAction(self.X_hat, local_coords_inv(eps)))
        C0 = numericalDifferential(opf, np.zeros((18, 1)))

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
        self.assertMatricesEqual(S1.t, S2.t)
        self.assertMatricesEqual(S1.b, S2.b)

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

    def continuousTimeDinamics(self, xi: State, U: InputSpace) -> np.ndarray:
        dot_T = SE23.vee(xi.T.as_matrix() @ (x_o(U.as_wa_mat()) - x_o(SE3.wedge(xi.b)) + D) + (G - D))
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
            f2 = lambda dt: (X * SymGroup.exp(continuous_lift(stateAction(X,xi), velocityAction(X,U)) * dt) * X.inv()).vec()
            Df2 = numericalDifferential(f2, 0.0)
            self.assertMatricesEqual(f1, Df2)

    def test_coords(self):
        for _ in range(self.test_reps):
            eps = 0.1*np.random.randn(18, 1)
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
    print("Lift is well defined!")
    print("*******************************")

    print("*******************************")
    print("Running coordinates lift test!")
    print("*******************************")
    test.test_coords()
    print("*******************************")
    print("Coords are fine!")
    print("*******************************")
