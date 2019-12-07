import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "Cartpole"

        # TODO: properly define the variables below
        self._action = None
        self._reward = 1
        self._isEnd = 0
        self._gamma = 1
        self._L = 20

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        self._failAngle = np.pi / 12  # fail angle
        self._boundaries = (-3, 3)  # cart boundaries
        self._Fmag = 10  # max motor force magnitude

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0  # total time elapsed  NOTE: you must use this variable

        self._state = np.array([self._x, self._v, self._theta, self._dtheta])

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return bool(self._isEnd)

    @property
    def state(self) -> np.ndarray:
        return self._state

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        # the state space equations of motion are:
        [x,v,theta,omega] = state
        self._action = action

        _omega = omega
        _F = -self._Fmag if action == 0 else self._Fmag  # force (action)

        _dx = v
        _domega = (self._g * np.sin(theta) + np.cos(theta) * (
                    (-_F - self._mp * self._l * (_omega ** 2) * np.sin(theta)) / (self._mc + self._mp))) / (
                              self._l * (4 / 3 - (self._mp * (np.cos(theta) ** 2)) / (self._mc + self._mp)))
        _dv = (_F + self._mp * self._l * (omega ** 2 * np.sin(theta) - _domega * np.cos(theta))) / (
                    self._mc + self._mp)
        _dtheta = _omega

        _dX = [_dx, _dv, _dtheta, _domega]

        #########
        # print("state::",state)
        # print("action::",action)
        _X = state
        self._state = state
        # print("dt::",self._dt)
        # print("dx::",_dX)
        # print(np.multiply(self._dt, _dX))
        # print(np.array(_X) + np.multiply(self._dt, _dX))
        return np.array(_X) + np.multiply(self._dt, _dX)

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        return 1#0 if self.isEnd == True else 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        self._isEnd = 1 if self.terminal() else 0
        # print("action:",action)
        # _X = np.array([self._x, self._v, self._theta, self._dtheta])


        if self._isEnd:
            self.reset()

        self._action = action
        self._t = self._t + self._dt


        nextState = self.nextState(self.state, action)
        self._reward = self.R(self.state, action, nextState)
        self._state = nextState.tolist()

        self._x = self.state[0]  # horizontal position of cart
        self._v = self.state[1]  # horizontal velocity of the cart
        self._theta = self.state[2]  # angle of the pole
        self._dtheta = self.state[3]  # angular velocity of the pole

        self._isEnd = 1 if self.terminal() else 0
        if self.terminal():
            self._isEnd = 1

        # print("t:", self._t, "action:", self._action, "reward:", self._reward)
        # print("isend:::::",self._isEnd)
        # print("newState:::",self._state)
        return (nextState, self.reward, self.isEnd)

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        self._action = None
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole
        # self._reward = 0
        pass

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        # print("terminal::theta::::",self._theta)
        if self._failAngle < abs(self._theta) or self._t > self._L or abs(self._x) > 3:
            return True
        else:
            return False

#     def test(self):
#         pass
#
#
# # Test
# c = Cartpole()
# c.test()
# while not c.isEnd:
#     c.step(action=1)