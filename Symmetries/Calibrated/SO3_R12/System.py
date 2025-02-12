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
    R: SO3 = SO3.identity()
    v: np.ndarray = np.zeros((3, 1))
    p: np.ndarray = np.zeros((3, 1))
    b: np.ndarray = np.zeros((6, 1))

    def __mul__(self, other) -> 'State':  # Define group multiplication
        assert (isinstance(other, State))
        return State(self.R * other.R,
                        self.v + other.v,
                        self.p + other.p,
                        self.b + other.b)

    @staticmethod
    def random():  # Create a random group element
        return State(SO3.exp(np.random.randn(3, 1)), np.random.randn(3, 1), np.random.randn(3, 1),
                        np.random.randn(6, 1))

    @staticmethod
    def identity():
        return State(SO3.identity(), np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((6, 1)))

    def inv(self) -> 'State':
        return State(self.R.inv(), -self.v, -self.p, -self.b)

    def exp(groupArray: np.ndarray) -> 'State':
        if not isinstance(groupArray, np.ndarray):
            raise TypeError
        elif not groupArray.shape == (15, 1):
            raise ValueError
        result = State()
        result.R = SO3.exp(groupArray[0:3, 0:1])
        result.v = groupArray[3:6, 0:1]
        result.p = groupArray[6:9, 0:1]
        result.b = groupArray[9:15, 0:1]
        return result

    def log(groupElement: 'State') -> np.ndarray:
        if not isinstance(groupElement, State):
            raise TypeError
        result = np.zeros((15, 1))
        result[0:3, 0:1] = SO3.log(groupElement.R)
        result[3:6, 0:1] = groupElement.v
        result[6:9, 0:1] = groupElement.p
        result[9:15, 0:1] = groupElement.b
        return result


# data is expected as (R, p, v, bw, ba, cal)
def stateFromData(d) -> "State":
    result = State()
    result.R = SO3(d[0])
    result.v = d[1]
    result.p = d[2]
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
