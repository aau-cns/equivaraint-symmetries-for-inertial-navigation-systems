import numpy as np
from dataclasses import dataclass
from pylie import SE3, SE23, SO3

# gravity
g = np.zeros((3, 1))
g[2] = -9.81
G = np.zeros((5, 5))
G[2, 3] = g[2]

# D matrix
D = np.zeros((5, 5))
D[3, 4] = 1.0


@dataclass
class State:
    T: SE23 = SE23.identity()  # (R, v, p)
    b: np.ndarray = np.zeros((6, 1))  # [b_w; b_a]

    # def inv(self) -> 'State':
    #     return State(self.T.inv(), -self.b)

    @staticmethod
    def random():  # Create a random state
        return State(SE23.exp(np.random.randn(9, 1)), np.random.randn(6, 1))

    def vec(self) -> np.ndarray:
        return np.vstack((self.T.R().as_euler().reshape(3, 1), self.T.x().as_vector(), self.T.w().as_vector(), self.b))

    # Extract SE3 block from SE23
    def A(self) -> SE3:
        a = SE3(self.T.R(), self.T.x())
        return a


# data is expected as (R, p, v, bw, ba, cal)
def stateFromData(d) -> "State":
    result = State()
    result.T = SE23(d[0], d[1], d[2])
    result.b = np.vstack((d[3], d[4]))
    return result


@dataclass  # Dataclass for velocity group
class InputSpace:
    w: np.ndarray = np.zeros((3, 3))
    a: np.ndarray = np.zeros((3, 1))
    tau: np.ndarray = np.zeros((6, 1))

    @staticmethod
    def random():
        U_rand = InputSpace()
        U_rand.w = SO3.wedge(np.random.randn(3, 1))
        U_rand.a = np.random.randn(3, 1)
        U_rand.tau = np.zeros((6, 1))
        return U_rand

    def as_vector(self) -> np.ndarray:
        vecc = np.zeros((12, 1))
        vecc[0:3, 0:1] = SO3.vee(self.w)
        vecc[3:6, 0:1] = self.a
        vecc[6:12, 0:1] = self.tau
        return vecc

    def as_wa_mat(self) -> np.ndarray:
        result = np.zeros((4, 4))
        result[0:3, 0:3] = self.w
        result[0:3, 3:4] = self.a
        return result

    def as_wa_vec(self) -> np.ndarray:
        result = np.zeros((6, 1))
        result[0:3] = SO3.vee(self.w)
        result[3:6] = self.a
        return result


# Xi_0
xi_0 = State()


# Measurement function
def measurePos(xi: State) -> np.ndarray:
    return xi.T.w().as_vector()


# get input space from vector form
def input_from_vector(vec) -> "InputSpace":
    if not isinstance(vec, np.ndarray):
        raise TypeError
    if not (vec.shape == (12, 1) or vec.shape == (6, 1)):
        raise ValueError
    result = InputSpace()
    result.w = SO3.wedge(vec[0:3, 0:1])
    result.a = vec[3:6, 0:1]
    if vec.shape == (12, 1):
        result.tau = vec[6:12, 0:1]
    return result
