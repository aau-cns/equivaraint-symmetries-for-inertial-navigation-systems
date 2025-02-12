import numpy as np
from dataclasses import dataclass
from pylie import SE3, SO3

# gravity
G = np.zeros((4, 4))
G[2, 3:4] = -9.81


@dataclass
class State:
    T: SE3 = SE3.identity()  # (R, v)
    b: np.ndarray = np.zeros((6, 1))  # [b_w; b_a]
    r: np.ndarray = np.zeros((3, 1))  # (r) - position of the sensor in world
    t: np.ndarray = np.zeros((3, 1))  # (t) - position of the sensor in body

    @staticmethod
    def random():  # Create a random state
        return State(SE3.exp(np.random.randn(6, 1)), np.random.randn(6, 1), np.random.randn(3, 1), np.random.randn(3, 1))

    def vec(self) -> np.ndarray:
        return np.vstack((self.T.R().as_euler().reshape(3, 1), self.T.x().as_vector(), self.b, self.r, self.t))


# Data is assumed to include (R, v, p, bw, ba, cal)
def stateFromData(d) -> "State":
    result = State()
    result.T = SE3(d[0], d[1])
    result.b = np.vstack((d[3], d[4]))
    result.t = d[5]
    result.r = d[2] + d[0] @ d[5] # p+Rt
    return result


@dataclass  # Dataclass for velocity group
class InputSpace:
    w: np.ndarray = np.zeros((3, 3))
    a: np.ndarray = np.zeros((3, 1))
    tau: np.ndarray = np.zeros((6, 1))
    mu: np.ndarray = np.zeros((3, 1))
    k: np.ndarray = np.zeros((3, 1))
    eta: np.ndarray = np.zeros((3, 1))

    @staticmethod
    def random():
        U_rand = InputSpace()
        U_rand.w = SO3.wedge(np.random.randn(3, 1))
        U_rand.a = np.random.randn(3, 1)
        U_rand.tau = np.zeros((6, 1))
        U_rand.mu = np.zeros((3, 1))
        U_rand.k = np.zeros((3, 1))
        U_rand.eta = np.zeros((3, 1))
        return U_rand

    def as_vector(self) -> np.ndarray:
        vecc = np.zeros((21, 1))
        vecc[0:3, 0:1] = SO3.vee(self.w)
        vecc[3:6, 0:1] = self.a
        vecc[6:12, 0:1] = self.tau
        vecc[12:15, 0:1] = self.mu
        vecc[15:18, 0:1] = self.k
        vecc[18:21, 0:1] = self.eta
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
    return xi.t


# get input space from vector form
def input_from_vector(vec) -> "InputSpace":
    if not isinstance(vec, np.ndarray):
        raise TypeError
    if not (vec.shape == (21, 1) or vec.shape == (6, 1)):
        raise ValueError
    result = InputSpace()
    result.w = SO3.wedge(vec[0:3, 0:1])
    result.a = vec[3:6, 0:1]
    if vec.shape == (21, 1):
        result.tau = vec[6:12, 0:1]
        result.mu = vec[12:15, 0:1]
        result.k = vec[15:18, 0:1]
        result.eta = vec[18:21, 0:1]
    return result
