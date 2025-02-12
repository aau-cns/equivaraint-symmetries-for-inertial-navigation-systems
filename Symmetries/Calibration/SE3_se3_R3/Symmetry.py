from Symmetries.Calibration.SE3_se3_R3.System import *
from Utils.matrix_math import *

# Coordinate selection, by default normal coordinates will be used,
# if exponential coordinates are selected remember to not use C_star
exponential_coords = False
fake_exponential = True


@dataclass  # Dataclass for symmetry group (SE(3) xx se(3)) x R(3) with calibration states
class SymGroup:
    B: SE3 = SE3.identity()
    beta: np.ndarray = np.zeros((4, 4))
    z: np.ndarray = np.zeros((3, 1))
    gamma: np.ndarray = np.zeros((3, 3))


    def __mul__(self, other) -> 'SymGroup':  # Define group multiplication
        assert (isinstance(other, SymGroup))
        return SymGroup(self.B * other.B,
                        self.beta + self.B.as_matrix() @ other.beta @ self.B.inv().as_matrix(),
                        self.z + other.z,
                        self.gamma + self.B.R().as_matrix() @ other.gamma @ self.B.R().inv().as_matrix())

    @staticmethod
    def random():  # Create a random group element
        b = SE3.wedge(np.random.randn(6, 1))
        g = SO3.wedge(np.random.randn(3, 1))
        return SymGroup(SE3.exp(np.random.randn(6, 1)),
                        b,
                        np.random.randn(3, 1),
                        g)

    @staticmethod
    def identity():
        return SymGroup(SE3.identity(),
                        np.zeros((4, 4)),
                        np.zeros((3, 1)),
                        np.zeros((3, 3)))

    def inv(self) -> 'SymGroup':
        internal_b_mat = -self.B.inv().as_matrix() @ self.beta @ self.B.as_matrix()
        internal_g_mat = -self.B.R().inv().as_matrix() @ self.gamma @ self.B.R().as_matrix()
        return SymGroup(self.B.inv(),
                        internal_b_mat,
                        -self.z,
                        internal_g_mat)

    def exp(groupArray: np.ndarray) -> 'SymGroup':
        if not isinstance(groupArray, np.ndarray):
            raise TypeError
        elif not groupArray.shape == (18, 1):
            raise ValueError
        result = SymGroup()
        result.B = SE3.exp(groupArray[0:6, 0:1])
        if fake_exponential:
            result.beta = SE3.wedge(groupArray[6:12, 0:1])
            result.gamma = SO3.wedge(groupArray[15:18, 0:1])
        else:
            result.beta = SE3.wedge(SE3LeftJacobian(groupArray[0:6, 0:1]) @ groupArray[6:12, 0:1])
            result.gamma = SO3.wedge(SO3LeftJacobian(groupArray[0:3, 0:1]) @ groupArray[15:18, 0:1])
        result.z = groupArray[12:15, 0:1]

        return result

    def log(groupElement: 'SymGroup') -> np.ndarray:
        if not isinstance(groupElement, SymGroup):
            raise TypeError
        result = np.zeros((18, 1))
        result[0:6, 0:1] = SE3.log(groupElement.B)
        if fake_exponential:
            result[6:12, 0:1] = SE3.vee(groupElement.beta)
            result[15:18, 0:1] = SO3.vee(groupElement.gamma)
        else:
            result[6:12, 0:1] = np.linalg.inv(SE3LeftJacobian(result[0:6, 0:1])) @ SE3.vee(groupElement.beta)
            result[15:18, 0:1] = np.linalg.inv(SO3LeftJacobian(result[0:3, 0:1])) @ SO3.vee(groupElement.gamma)
        result[12:15, 0:1] = groupElement.z
        return result

    def vec(self):
        return np.vstack((self.B.R().as_euler().reshape(3, 1),
                          self.B.x().as_vector(),
                          SE3.vee(self.beta),
                          self.z,
                          SO3.vee(self.gamma)))


def SO3LeftJacobian(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError
    elif not arr.shape == (3, 1):
        raise ValueError

    angle = np.linalg.norm(arr)

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(angle, 0.):
        return np.eye(3) + 0.5 * SO3.wedge(arr)

    axis = arr / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return (s / angle) * np.eye(3) + \
           (1 - s / angle) * np.outer(axis, axis) + \
           ((1 - c) / angle) * SO3.wedge(axis)


def SE3LeftJacobianQ(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError
    elif not arr.shape == (6, 1):
        raise ValueError

    phi = arr[0:3, 0:1]  # rotation part
    rho = arr[3:6, 0:1]  # translation part

    rx = SO3.wedge(rho)
    px = SO3.wedge(phi)

    ph = np.linalg.norm(phi)
    ph2 = ph * ph
    ph3 = ph2 * ph
    ph4 = ph3 * ph
    ph5 = ph4 * ph

    cph = np.cos(ph)
    sph = np.sin(ph)

    m1 = 0.5
    m2 = (ph - sph) / ph3
    m3 = (0.5 * ph2 + cph - 1.) / ph4
    m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

    t1 = rx
    t2 = px @ rx + rx @ px + px @ rx @ px
    t3 = px @ px @ rx + rx @ px @ px - 3. * px @ rx @ px
    t4 = px @ rx @ px @ px + px @ px @ rx @ px

    return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4


def SE3LeftJacobian(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError
    elif not arr.shape == (6, 1):
        raise ValueError

    phi = arr[0:3, 0:1]  # rotation part
    rho = arr[3:6, 0:1]  # translation part

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(np.linalg.norm(phi), 0.):
        return np.eye(6) + 0.5 * SE3.adjoint(arr)

    SO3_JL = SO3LeftJacobian(phi)
    QL = SE3LeftJacobianQ(arr)

    J = np.zeros([6, 6])
    J[0:3, 0:3] = SO3_JL
    J[0:3, 3:6] = QL
    J[3:6, 3:6] = SO3_JL

    return J

# adjoint matrix
def grp_adj(l: np.ndarray) -> np.ndarray:
    if not (l.shape == (18, 1)):
        raise ValueError
    ad = np.zeros((18, 18))
    ad[0:6, 0:6] = SE3.adjoint(l[0:6])
    ad[0:6, 6:12] = SE3.adjoint(l[6:12])
    ad[6:12, 6:12] = SE3.adjoint(l[0:6])
    ad[15:18, 0:3] = SO3.wedge(l[15:18])
    ad[15:18, 15:18] = SO3.wedge(l[0:3])
    return ad


def Adj_action(X: SymGroup, x: np.ndarray) -> np.ndarray:
    L1 = SE3.wedge(x[0:6, :])
    L2 = x[6:12, :]
    L3 = x[12:15, :]
    L4 = x[15:18, :]
    result = np.zeros((18, 1))
    result[0:6, :] = SE3.vee(X.B.as_matrix() @ L1 @ X.B.inv().as_matrix())
    result[6:12, :] = X.B.Adjoint() @ L2 - SE3.adjoint(X.B.Adjoint() @ x[0:6, :]) @ SE3.vee(X.beta)
    result[12:15, :] = L3
    result[15:18, :] = X.B.R().as_matrix() @ L3 - SO3.wedge(X.B.R().as_matrix() @ x[0:3, :]) @ SO3.vee(X.gamma)
    return result


def stateAction(X: SymGroup, xi: State) -> State:
    internal_b = X.B.inv().Adjoint() @ (xi.b - SE3.vee(X.beta))
    internal_d = X.B.R().inv().as_matrix() @ (xi.t - SO3.vee(X.gamma))
    return State(xi.T * X.B, internal_b, xi.r + X.z,  internal_d)


def velocityAction(X: SymGroup, U: InputSpace) -> InputSpace:
    result = InputSpace()
    Ad = X.B.inv().Adjoint() @ (U.as_wa_vec() - SE3.vee(X.beta))
    result.w = SO3.wedge(Ad[0:3, 0:1])
    result.a = Ad[3:6, 0:1]
    result.mu = X.B.R().inv().as_matrix() @ (U.mu - X.B.x().as_vector())
    result.tau = X.B.inv().Adjoint() @ U.tau
    result.k = X.B.R().inv().as_matrix() @ (U.k + SO3.vee(X.gamma))
    result.eta = X.B.R().inv().as_matrix() @ U.eta
    return result


def outputAction(X: SymGroup, y: np.ndarray) -> np.ndarray:
    return y + X.z


def local_coords(e: State) -> np.ndarray:
    theta_e = SymGroup()
    theta_e.B = xi_0.T.inv() * e.T
    theta_e.z = e.r - xi_0.r
    if exponential_coords:
        theta_e.beta = e.b - xi_0.b
        theta_e.gamma = e.t - xi_0.t
        eps = np.vstack((SE3.log(theta_e.B), theta_e.beta, theta_e.z, theta_e.gamma))
    else:
        theta_e.beta = SE3.wedge(xi_0.b - SE3.Adjoint(theta_e.B) @ e.b)
        theta_e.gamma = SO3.wedge(xi_0.t - theta_e.B.R().as_matrix() @ e.t)
        eps = SymGroup.log(theta_e)

    return eps


def local_coords_inv(eps: np.ndarray) -> "State":
    if exponential_coords:
        inv_theta_eps = State()
        inv_theta_eps.T = xi_0.T * SE3.exp(eps[0:6, 0:1])
        inv_theta_eps.b = xi_0.b + eps[6:12, 0:1]
        inv_theta_eps.r = xi_0.r + eps[12:15, 0:1]
        inv_theta_eps.t = xi_0.t + eps[15:18, 0:1]
    else:
        inv_theta_eps = stateAction(SymGroup.exp(eps), xi_0)

    return inv_theta_eps


# (D\theta) * (D\phi_{xi}(E) in E = Id)
def stateActionDiff(xi: State) -> np.ndarray:
    coordsAction = lambda U: local_coords(stateAction(SymGroup.exp(U), xi))
    differential = numericalDifferential(coordsAction, np.zeros((18, 1)))
    return differential
