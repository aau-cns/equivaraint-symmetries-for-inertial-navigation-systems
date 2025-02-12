from Symmetries.Calibrated.SE23_se23.System import *
from dataclasses import dataclass
from pylie import SE23, SO3
from Utils.matrix_math import *

# Coordinate selection, by default normal coordinates will be used,
# if exponential coordinates are selected remember to not use C_star
exponential_coords = False
fake_exponential = False


@dataclass  # Dataclass for symmetry group SE2(3) xx se2(3)
class SymGroup:
    B: SE23 = SE23.identity()
    beta: np.ndarray = np.zeros((5, 5))

    def __mul__(self, other) -> 'SymGroup':  # Define group multiplication
        assert (isinstance(other, SymGroup))
        return SymGroup(self.B * other.B, self.beta + self.B.as_matrix() @ other.beta @ self.B.inv().as_matrix())

    @staticmethod
    def random():  # Create a random group element
        return SymGroup(SE23.exp(np.random.randn(9, 1)), SE23.wedge(np.random.randn(9, 1)))

    @staticmethod
    def identity():
        return SymGroup(SE23.identity(), np.zeros((5, 5)))

    def inv(self) -> 'SymGroup':
        return SymGroup(self.B.inv(), -self.B.inv().as_matrix() @ self.beta @ self.B.as_matrix())

    def exp(groupArray: np.ndarray) -> 'SymGroup':
        if not isinstance(groupArray, np.ndarray):
            raise TypeError
        elif not groupArray.shape == (18, 1):
            raise ValueError
        result = SymGroup()
        if fake_exponential:
            result.beta = SE23.wedge(groupArray[9:18, 0:1])
        else:
            result.beta = SE23.wedge(SE23LeftJacobian(groupArray[0:9, 0:1]) @ groupArray[9:18, 0:1])
        result.B = SE23.exp(groupArray[0:9, 0:1])
        return result

    def log(groupElement: 'SymGroup') -> np.ndarray:
        if not isinstance(groupElement, SymGroup):
            raise TypeError
        result = np.zeros((18, 1))
        result[0:9, 0:1] = SE23.log(groupElement.B)
        if fake_exponential:
            result[9:18, 0:1] = SE23.vee(groupElement.beta)
        else:
            result[9:18, 0:1] = np.linalg.inv(SE23LeftJacobian(result[0:9, 0:1])) @ SE23.vee(groupElement.beta)
        return result

    def vec(self):
        return np.vstack(
            (self.B.R().as_euler().reshape(3, 1), self.B.x().as_vector(), self.B.w().as_vector(), SE23.vee(self.beta)))


def J1(so3vec: np.ndarray) -> np.ndarray:
    if not isinstance(so3vec, np.ndarray):
        raise TypeError
    elif not so3vec.shape == (3, 1):
        raise ValueError

    angle = np.linalg.norm(so3vec)

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(angle, 0.0):
        print("Using small angle approximation for J1")
        return np.eye(3) + 0.5 * SO3.wedge(so3vec)

    axis = so3vec / angle
    s = np.sin(angle) / angle
    c = (1 - np.cos(angle)) / angle

    return s * np.eye(3) + (1 - s) * np.outer(axis, axis) + c * SO3.wedge(axis)


def Q1(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError
    elif not arr.shape == (6, 1):
        raise ValueError

    phi = arr[0:3, 0:1]  # rotation part
    rho = arr[3:6, 0:1]  # velocity/translation part

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
    m3 = (0.5 * ph2 + cph - 1.0) / ph4
    m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

    t1 = rx
    t2 = px @ rx + rx @ px + px @ rx @ px
    t3 = px @ px @ rx + rx @ px @ px - 3.0 * px @ rx @ px
    t4 = px @ rx @ px @ px + px @ px @ rx @ px

    return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

def SO3LeftJacobian(so3vec: np.ndarray) -> np.ndarray:
    return J1(so3vec)


def SE3LeftJacobian(se3vec: np.ndarray) -> np.ndarray:
    if not isinstance(se3vec, np.ndarray):
        raise TypeError
    elif not se3vec.shape == (6, 1):
        raise ValueError

    phi = se3vec[0:3, 0:1]  # rotation part
    # rho = se3vec[3:6, 0:1]  # translation part

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(np.linalg.norm(phi), 0.0):
        return np.eye(6) + 0.5 * SE3.adjoint(se3vec)

    SO3_JL = J1(phi)

    J = np.zeros([6, 6])
    J[0:3, 0:3] = SO3_JL
    J[3:6, 3:6] = SO3_JL
    J[3:6, 0:3] = Q1(se3vec)

    return J


def SE23LeftJacobian(se23vec: np.ndarray) -> np.ndarray:
    if not isinstance(se23vec, np.ndarray):
        raise TypeError
    elif not se23vec.shape == (9, 1):
        raise ValueError

    phi = se23vec[0:3, 0:1]  # rotation part
    rho = se23vec[3:6, 0:1]  # translation part 1
    psi = se23vec[6:9, 0:1]  # translation part 2

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(np.linalg.norm(phi), 0.0):
        return np.eye(9) + 0.5 * SE23.adjoint(se23vec)

    SO3_JL = J1(phi)

    J = np.zeros([9, 9])
    J[0:3, 0:3] = SO3_JL
    J[3:6, 3:6] = SO3_JL
    J[6:9, 6:9] = SO3_JL
    J[3:6, 0:3] = Q1(np.vstack((phi, rho)))
    J[6:9, 0:3] = Q1(np.vstack((phi, psi)))

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
    if not (l.shape == (18, 1)):
        raise ValueError
    ad = np.zeros((18, 18))
    ad[0:9, 0:9] = SE23.adjoint(l[0:9])
    ad[9:18, 0:9] = SE23.adjoint(l[9:18])
    ad[9:18, 9:18] = SE23.adjoint(l[0:9])
    return ad


def stateAction(X: SymGroup, xi: State) -> State:
    return State(xi.T * X.B, SE23.vee(X.B.inv().as_matrix() @ (SE23.wedge(xi.b) - X.beta) @ X.B.as_matrix()))


def velocityAction(X: SymGroup, U: InputSpace) -> InputSpace:
    result_vec = np.zeros((18, 1))
    result_vec[0:9, 0:1] = SE23.vee(
        X.B.inv().as_matrix() @ (U.as_W_mat() - X.beta) @ X.B.as_matrix() + f_10(X.B.inv().as_matrix()))
    result_vec[9:18, 0:1] = SE23.vee(X.B.inv().as_matrix() @ SE23.wedge(U.tau) @ X.B.as_matrix())
    return input_from_vector(result_vec)


def local_coords(e: State) -> np.ndarray:
    theta_e = SymGroup()
    theta_e.B = xi_0.T.inv() * e.T
    if exponential_coords:
        theta_e.beta = e.b - xi_0.b
        eps = np.vstack((SE23.log(theta_e.B), theta_e.beta))
    else:
        theta_e.beta = SE23.wedge(xi_0.b) - theta_e.B.as_matrix() @ SE23.wedge(e.b) @ theta_e.B.inv().as_matrix()
        eps = SymGroup.log(theta_e)
    return eps


def local_coords_inv(eps: np.ndarray) -> "State":
    if exponential_coords:
        inv_theta_eps = State()
        inv_theta_eps.T = xi_0.T * SE23.exp(eps[0:9, 0:1])
        inv_theta_eps.b = xi_0.b + eps[9:18, 0:1]
    else:
        inv_theta_eps = stateAction(SymGroup.exp(eps), xi_0)
    return inv_theta_eps


# (D\theta) * (D\phi_{xi}(E) in E = Id)
def stateActionDiff(xi: State) -> np.ndarray:
    coordsAction = lambda U: local_coords(stateAction(SymGroup.exp(U), xi))
    differential = numericalDifferential(coordsAction, np.zeros((18, 1)))
    return differential
