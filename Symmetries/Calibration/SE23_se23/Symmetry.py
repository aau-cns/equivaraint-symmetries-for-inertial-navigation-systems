from Symmetries.Calibration.SE23_se23.System import *
from dataclasses import dataclass
from pylie import SE23, SO3
from Utils.matrix_math import *

# Coordinate selection, by default normal coordinates will be used,
# if exponential coordinates are selected remember to not use C_star
exponential_coords = False
fake_exponential = True


@dataclass  # Dataclass for symmetry group SE2(3) xx se2(3) with calibration states
class SymGroup:
    B: SE23 = SE23.identity()
    beta: np.ndarray = np.zeros((5, 5))
    gamma: np.ndarray = np.zeros((3, 3))

    def __mul__(self, other) -> 'SymGroup':  # Define group multiplication
        assert (isinstance(other, SymGroup))
        return SymGroup(self.B * other.B,
                        self.beta + self.B.as_matrix() @ other.beta @ self.B.inv().as_matrix(),
                        self.gamma + self.B.R().as_matrix() @ other.gamma @ self.B.R().inv().as_matrix())

    @staticmethod
    def random():  # Create a random group element
        return SymGroup(SE23.exp(np.random.randn(9, 1)),
                        SE23.wedge(np.random.randn(9, 1)),
                        SO3.wedge(np.random.randn(3, 1)))

    @staticmethod
    def identity():
        return SymGroup(SE23.identity(),
                        np.zeros((5, 5)),
                        np.zeros((3, 3)))

    def inv(self) -> 'SymGroup':
        return SymGroup(self.B.inv(),
                        -self.B.inv().as_matrix() @ self.beta @ self.B.as_matrix(),
                        -self.B.R().inv().as_matrix() @ self.gamma @ self.B.R().as_matrix())

    def exp(groupArray: np.ndarray) -> 'SymGroup':
        if not isinstance(groupArray, np.ndarray):
            raise TypeError
        elif not groupArray.shape == (21, 1):
            raise ValueError
        result = SymGroup()
        if fake_exponential:
            result.beta = SE23.wedge(groupArray[9:18, 0:1])
            result.gamma = SO3.wedge(groupArray[18:21, 0:1])
        else:
            result.beta = SE23.wedge(SE23LeftJacobian(groupArray[0:9, 0:1]) @ groupArray[9:18, 0:1])
            result.gamma = SO3.wedge(SO3LeftJacobian(groupArray[0:3, 0:1]) @ groupArray[18:21, 0:1])
        result.B = SE23.exp(groupArray[0:9, 0:1])
        return result

    def log(groupElement: 'SymGroup') -> np.ndarray:
        if not isinstance(groupElement, SymGroup):
            raise TypeError
        result = np.zeros((21, 1))
        result[0:9, 0:1] = SE23.log(groupElement.B)
        if fake_exponential:
            result[9:18, 0:1] = SE23.vee(groupElement.beta)
            result[18:21, 0:1] = SO3.vee(groupElement.gamma)
        else:
            result[9:18, 0:1] = np.linalg.inv(SE23LeftJacobian(result[0:9, 0:1])) @ SE23.vee(groupElement.beta)
            result[18:21, 0:1] = np.linalg.inv(SO3LeftJacobian(result[0:3, 0:1])) @ SO3.vee(groupElement.gamma)
        return result

    def vec(self):
        return np.vstack((self.B.R().as_euler().reshape(3, 1),
                          self.B.x().as_vector(),
                          self.B.w().as_vector(),
                          SE23.vee(self.beta),
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


def SE23LeftJacobian(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError
    elif not arr.shape == (9, 1):
        raise ValueError

    phi = arr[0:3, 0:1]  # rotation part
    rho = arr[3:6, 0:1]  # translation part 1
    psi = arr[6:9, 0:1]  # translation part 2

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(np.linalg.norm(phi), 0.):
        return np.eye(9) + 0.5 * SE23.adjoint(arr)

    SO3_JL = SO3LeftJacobian(phi)
    QL1 = SE3LeftJacobianQ(np.vstack((phi, rho)))
    QL2 = SE3LeftJacobianQ(np.vstack((phi, psi)))

    J = np.zeros([9, 9])
    J[0:3, 0:3] = SO3_JL
    J[3:6, 3:6] = SO3_JL
    J[6:9, 6:9] = SO3_JL
    J[0:3, 3:6] = QL1
    J[0:3, 6:9] = QL2

    return J


# From (R,v,p) to (0,0,v)
def f_10(mat: np.ndarray) -> np.ndarray:
    if not (mat.shape == (5, 5)):
        raise ValueError
    f = np.zeros((5, 5))
    f[0:3, 4:5] = mat[0:3, 3:4]
    return f

# adjoint matrix
def grp_adj(l: np.ndarray) -> np.ndarray:
    if not (l.shape == (21, 1)):
        raise ValueError
    ad = np.zeros((21, 21))
    ad[0:9, 0:9] = SE23.adjoint(l[0:9])
    ad[9:18, 0:9] = SE23.adjoint(l[9:18])
    ad[9:18, 9:18] = SE23.adjoint(l[0:9])
    ad[18:21, 0:3] = SO3.wedge(l[18:21])
    ad[18:21, 18:21] = SO3.wedge(l[0:3])
    return ad


def Adj_action(X: SymGroup, x: np.ndarray) -> np.ndarray:
    L1 = SE23.wedge(x[0:9, :])
    L2 = x[9:18, :]
    L3 = x[18:21, :]
    result = np.zeros((21, 1))
    result[0:9, :] = SE23.vee(X.B.as_matrix() @ L1 @ X.B.inv().as_matrix())
    result[9:18, :] = X.B.Adjoint() @ L2 - SE23.adjoint(X.B.Adjoint() @ x[0:9, :]) @ SE23.vee(X.beta)
    result[18:21, :] = X.B.R().as_matrix() @ L3 - SO3.wedge(X.B.R().as_matrix() @ x[0:3, :]) @ SO3.vee(X.gamma)
    return result


def stateAction(X: SymGroup, xi: State) -> State:
    return State(xi.T * X.B,
                 SE23.vee(X.B.inv().as_matrix() @ (SE23.wedge(xi.b) - X.beta) @ X.B.as_matrix()),
                 X.B.R().inv().as_matrix() @ (xi.t - SO3.vee(X.gamma)))


def velocityAction(X: SymGroup, U: InputSpace) -> InputSpace:
    result_vec = np.zeros((21, 1))
    result_vec[0:9, 0:1] = SE23.vee(
        X.B.inv().as_matrix() @ (U.as_W_mat() - X.beta) @ X.B.as_matrix() + f_10(X.B.inv().as_matrix()))
    result_vec[9:18, 0:1] = SE23.vee(X.B.inv().as_matrix() @ SE23.wedge(U.tau) @ X.B.as_matrix())
    result_vec[18:21, 0:1] = X.B.R().inv().as_matrix() @ U.eta
    return input_from_vector(result_vec)


def outputAction(X: SymGroup, y: np.ndarray) -> np.ndarray:
    return X.B.R().inv().as_matrix() @ (y + SO3.vee(X.gamma) - X.B.w().as_vector())


def local_coords(e: State) -> np.ndarray:
    theta_e = SymGroup()
    theta_e.B = xi_0.T.inv() * e.T
    if exponential_coords:
        theta_e.beta = e.b - xi_0.b
        theta_e.gamma = e.t - xi_0.t
        eps = np.vstack((SE23.log(theta_e.B), theta_e.beta, theta_e.gamma))
    else:
        theta_e.beta = SE23.wedge(xi_0.b) - theta_e.B.as_matrix() @ SE23.wedge(e.b) @ theta_e.B.inv().as_matrix()
        theta_e.gamma = SO3.wedge(xi_0.t - theta_e.B.R().as_matrix() @ e.t)
        eps = SymGroup.log(theta_e)
    return eps


def local_coords_inv(eps: np.ndarray) -> "State":
    if exponential_coords:
        inv_theta_eps = State()
        inv_theta_eps.T = xi_0.T * SE23.exp(eps[0:9, 0:1])
        inv_theta_eps.b = xi_0.b + eps[9:18, 0:1]
        inv_theta_eps.t = xi_0.t + eps[18:21, 0:1]
    else:
        inv_theta_eps = stateAction(SymGroup.exp(eps), xi_0)
    return inv_theta_eps


# (D\theta) * (D\phi_{xi}(E) in E = Id)
def stateActionDiff(xi: State) -> np.ndarray:
    coordsAction = lambda U: local_coords(stateAction(SymGroup.exp(U), xi))
    differential = numericalDifferential(coordsAction, np.zeros((21, 1)))
    return differential
