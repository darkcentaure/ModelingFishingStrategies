"""
User-defined Python classes for Optimal Fishing Strategy Objects
Excluding Fishing Objects, thus only, Optimal Control Objects

Author: Niels in 't Veld
Created: 2/03/2022
Updated: 08/05/2022
Version: 3.0
"""
import time
from abc import abstractmethod, ABC
import numpy as np
from FishingPools import *
from scipy.optimize import fsolve


## TODO'S
# 1 Remove timeline from solutions

###
# General Functions
# Updated: 02/04/22
###
def generateTimeline(begin: float, end: float, steps: int) -> np.array:
    """ function that generates a discrete timeline from begin to end with n number of steps.
    Leading to step size: (end - begin) / steps."""
    # steps = n
    # length of array = n + 1
    stepSize = (end - begin) / steps
    array = np.array([i * stepSize for i in range(steps + 1)])
    return array


###
# Blueprints
###
class _ControlProblem:
    """
    BLUEPRINT OF A CONTROL PROBLEM

    Consider a optimal control problem of optimizing a functional, namely,
    J(x,u) = int_0^T L(x,u,t) dt + g(x(t)),
    where L and M are running and terminal costs, resp.
    subject to the state-equation, i.e.
    x'(t) = f(x,u) , x(0) = 0
    where x(t) and u(t) are state and control trajectories.

    The objective: maximizing_u {J(x,u), u measurable} = minimizing_u {-J(x,u), u measurable}

    This leads to the HBJ equations:
    x'(t) = f(x,u)
    -y'(t) = L_x(x,u) + f_x(x,y) * y(t)
    u(t) = argmin_u { L(x,u) + f(x,u) * y(t) }

    And the Hamiltonian System: H(x,y,u) =  L(x,u) + f(x,u) * y(t)
    x'(t) = H_y(x,y,u)
    y'(t) = - H_x(x,y,u)
    u(t) = argmin_u { H(x,y,u) }
    """
    begin = 0
    end = 1
    N = 200
    dt = (end - begin) / N
    timeline = generateTimeline(begin, end, N)

    # solutions part
    problemIsInitialized: bool = False
    problemHorizonIsAppended: bool = False
    solution = None
    solution_dictionary: dict = {}  # (iteration: int, Solution)
    iteration: int = 0
    error_list: list = []

    def adjustSteps(self, N: int, end=None):
        self.N = N
        if end is not None:
            self.end = end
        self.dt = (self.end - self.begin) / self.N
        self.timeline = generateTimeline(self.begin, self.end, self.N)
        self.problemIsInitialized = False
        print(self)
        return self

    def appendHorizon(self, dN: int):
        self.end = self.end + dN * self.dt
        self.adjustSteps(self.N + dN)
        self.problemHorizonIsAppended = True
        return self

    def generateErrorDictionary(self):
        self.error_list = []
        if len(self.solution_dictionary) > 1:
            for i in range(len(self.solution_dictionary) - 1):
                error = self.solution_dictionary[i] - self.solution_dictionary[i + 1]
                self.error_list.append(error)
            return self.error_list
        else:
            raise NotImplemented("Not enough solutions to create a dictionary")

    def plotConsecutiveSolutionsTrajectory(self,
                                           plot=("density" or "harvesting" or "dual"),
                                           plot_errors=False, title_name: str = "iteration"):
        if type(self.solution) is Solution1D:
            import matplotlib.pyplot as plt
            plt.Figure()

            plt.xlabel("time (t)")

            if not plot_errors:
                list_of_items = self.solution_dictionary.values()
                plt.ylabel(plot + " (t)")
            else:
                list_of_items = self.generateErrorDictionary()
                plt.ylabel("error in " + plot + " (t)")
            i = 0
            for item in list_of_items:
                array = np.zeros_like(item.timeline)

                if plot == "density":
                    array = item.state
                elif plot == "harvesting":
                    array = item.control
                elif plot == "dual":
                    array = item.costate
                else:
                    NameError("wrong input name")

                plt.plot(item.timeline, array, label=item.plot_name)  # TODO fix color bullshit
                plt.legend(title=title_name, loc=0)
                i += 1
            plt.show()
            pass
        else:
            raise NotImplemented

    def plotResiduals(self):
        if type(self.solution) is Solution1D:
            list_of_iterations = [i for i in range(self.iteration)]
            residual_list = [0 for i in range(self.iteration)]

            for key, value in self.solution_dictionary.items():
                residual_list[key] = value.residual
            pass
            import matplotlib.pyplot as plt
            plt.Figure()
            plt.xlabel("iteration")
            plt.ylabel("Residual")
            plt.plot(list_of_iterations, residual_list)
            plt.show()
            pass
        else:
            raise NotImplemented

    def __str__(self):
        string = "CP [" + str(self.begin) + ", " + str(self.end) + "],N=" + str(self.N)
        return string

    pass


###
# One Fish Species
# Updated: 08/05/22
###
class Solution1D:
    """
    Solution Class with different Trajectories

    Trajectory is a curve,
    X = {x_t: t=0, ..., T },
    with corresponding 'dual' trajectory,
    y:= Lambda = {lambda_t: t=0, ..., T}.
    and control
    U = {u_t for t=0, ..., T}.
    """
    residual = None
    plot_name: str = None  # I add this to make plots legends more explanatory
    functional_value = None
    '''value of the corresponding Functional'''

    def __init__(self, timeline: np.array, state: np.array, costate: np.array, control: np.array):
        self.timeline: np.array = timeline
        self.state: np.array = state
        self.costate: np.array = costate
        self.control: np.array = control
        pass

    def addName(self, name):
        self.plot_name = name
        return self

    def passY(self):
        return np.append(self.state[1:], self.costate[1:])

    def plotTrajectory(self, plot=("density" or "harvesting" or "dual")):
        array: np.array = np.zeros_like(self.timeline)
        if plot == "density":
            array = self.state
        elif plot == "harvesting":
            array = self.control
        elif plot == "dual":
            array = self.costate
        else:
            raise NameError("wrong input name")

        import matplotlib.pyplot as plt
        plt.Figure()
        plt.plot(self.timeline, array)
        plt.xlabel("time (t)")
        plt.ylabel(plot + " (t)")
        plt.title("n =" + str(len(array) - 1))
        plt.show()
        pass

    def plotSolution(self, show_plot=True):
        import matplotlib.pyplot as plt
        plt.subplots(nrows=2, ncols=1)
        plt.suptitle("N =" + str(len(self.timeline) - 1) + ", u=" + str(np.round(self.functional_value, 5)))
        plt.subplot(2, 1, 1).plot(self.timeline, self.state)
        plt.ylabel("state x(t)")
        plt.gca().get_xaxis().set_visible(False)
        plt.subplot(2, 1, 2).plot(self.timeline, self.control)
        plt.xlabel("time (t)")
        plt.ylabel("control a(t)")
        if show_plot is True:
            plt.show()
        pass

    def plotTrajectories(self, show=True):
        import matplotlib.pyplot as plt
        plt.subplots(nrows=3, ncols=1)
        plt.suptitle("N =" + str(len(self.timeline) - 1) + "u=" + str(np.round(self.functional_value, 5)))
        plt.subplot(3, 1, 1).plot(self.timeline, self.state)
        plt.ylabel("state x(t)")
        plt.gca().get_xaxis().set_visible(False)
        plt.subplot(3, 1, 2).plot(self.timeline, self.costate)
        plt.ylabel("costate y(t)")
        plt.gca().get_xaxis().set_visible(False)
        plt.subplot(3, 1, 3).plot(self.timeline, self.control)
        plt.xlabel("time (t)")
        plt.ylabel("control a(t)")
        if show is True:
            plt.show()
        pass

    def __mod__(self, other):
        return abs(self.residual - other.residual)

    def __sub__(self, other):
        return self.functional_value - self.functional_value

    def __le__(self, other):  # self <= other
        return self.functional_value <= other.functional_value

    pass


class _Error(Solution1D):  # TODO
    def __init__(self, timeline: np.array, state: np.array, costate: np.array, control: np.array):
        super().__init__(timeline, state, costate, control)
        pass

    def passY(self):
        raise NotImplemented

    def plotTrajectory(self, plot=("density" or "harvesting" or "dual")):
        array: np.array = np.zeros_like(self.timeline)
        if plot == "density":
            array = self.state
        elif plot == "harvesting":
            array = self.control
        elif plot == "dual":
            array = self.costate
        else:
            NameError("wrong input name")

        import matplotlib.pyplot as plt
        plt.Figure()
        plt.plot(self.timeline, array)
        plt.xlabel("time (t)")
        plt.ylabel("error in " + plot + " (t)")
        plt.title("n =" + str(len(array) - 1))
        plt.show()
        pass

    def plotTrajectories(self):
        # if not self.harvestingTrajectory.any(None):
        import matplotlib.pyplot as plt
        plt.subplots(nrows=3, ncols=1)
        plt.suptitle("n =" + str(len(self.timeline) - 1))
        plt.subplot(3, 1, 1).plot(self.timeline, self.state)
        plt.ylabel("error in x(t)")
        plt.gca().get_xaxis().set_visible(False)
        # TODO: remove axis
        plt.subplot(3, 1, 2).plot(self.timeline, self.costate)
        plt.ylabel("error in y(t)")
        plt.gca().get_xaxis().set_visible(False)
        plt.subplot(3, 1, 3).plot(self.timeline, self.control)
        plt.xlabel("time (t)")
        plt.ylabel("error in a(t)")
        plt.show()
        print(self.computeNorms())
        pass

    def computeNorms(self):
        state_error = np.linalg.norm(self.state, 2)
        costate_error = np.linalg.norm(self.costate, 2)
        control_error = np.linalg.norm(self.control, 2)
        return state_error, costate_error, control_error

    pass


class _SingleSpeciesProblem(_ControlProblem, ABC):
    """ BLUEPRINT...
    A Fishing Problem is a combination of a Fish Pool (a 'set' of specific parameters for the fish population)
    and a Control Problem (A 'set' of functions for solving the Fishing Problem)

    J(x,u) = int_0^T L(x,u) dt + g(x(t))

    L(x,u) = 'depends on situation'
    g(x(T)) = C (x(T) - x_0)^2
    """
    fp: SingleSpeciesPool = None
    '''a fish pool of parameters and related functions'''
    C = 1.0
    '''the weight of the final condition'''
    solution_dictionary: dict[int, Solution1D] = {}  # (iteration: int, Solution)
    '''importance of terminal cost'''

    def __init__(self, fishPool: SingleSpeciesPool = SingleSpeciesPool()):
        super().__init__()
        self.fp: SingleSpeciesPool = fishPool
        pass

    # CALCULATING
    def generateInitialCondition(self) -> np.array:
        """ Y = [x_0 ... x_0, 0 ... 0]  """
        guess = np.zeros(2 * self.N)
        guess[:self.N] = self.fp.x_0
        return guess

    def symplecticEulerF(self, Y: np.array) -> np.array:
        """
        x_{n+1} = x_n + t H_y (x_n, y_{n+1}),
        y_{n} = y_{n+1} + t H_x (x_n, y_{n+1})
        """
        FY: np.array = np.zeros_like(Y)

        N: int = self.N
        x, u = np.array_split(Y, 2)
        ''' x is state and u is corresponding costate'''

        # F(X_1) = -X_1 + X_0 + t H_lambda(x,y):
        FY[0] = -x[0] + self.fp.x_0 + self.dt * self.H_y(self.fp.x_0, u[0])

        FY[1:N] = -x[1:N] + x[:N - 1] + self.dt * self.H_y(x[:N - 1], u[1:N])
        FY[N:-1] = -u[:N - 1] + u[1:N] + self.dt * self.H_x(x[:N - 1], u[1:N])

        # F(lambda(T)) = -lambda(T) + g'(X(T)):
        FY[-1] = -u[-1] + self.C * self.g_x(x[-1])
        return FY

    def solve(self, print_residual: bool = False) -> np.array:
        """
        F(Y) = 0 via fsolve, where Y = (x, lambda)
        """
        if self.problemIsInitialized:
            guess = self.solution.passY()
        else:
            guess = self.generateInitialCondition()
            self.problemIsInitialized = True

        sol = fsolve(func=self.symplecticEulerF, x0=guess)
        # TODO: fsolve should also be able to give the residual output directly
        residual = sum(abs(self.symplecticEulerF(sol)))
        if print_residual:
            print("residual", residual)

        self.updateSolutionAndAddToDictionary(sol, residual)
        self.iteration += 1
        return self.solution

    def updateSolutionAndAddToDictionary(self, Y, residual):
        state, costate = np.split(Y, 2)
        state = np.insert(state, 0, self.fp.x_0)

        initial_costate = costate[0] + self.dt * self.H_x(state[0], costate[0])
        costate = np.insert(costate, 0, initial_costate)

        self.solution: Solution1D = Solution1D(self.timeline, state, costate,
                                               self.computeOptimalControl(state, costate)).addName(str(self.iteration))
        self.solution.residual = residual
        functional_value = self.functional(state, self.computeOptimalControl(state, costate))
        # print('J:', functional_value)
        self.solution.functional_value = functional_value
        self.solution_dictionary[self.iteration] = self.solution
        pass

    # PLOTTING
    def plotDictionary(self, show: bool):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2, ncols=1)

        plt.suptitle("n =" + str(len(self.timeline) - 1))

        for item in self.solution_dictionary.items():
            key, value = item
            ax[0].plot(value.timeline, value.state, label=key)
            ax[1].plot(value.timeline, value.control, label=key)
            # TODO: make key in legend
            pass
        ax[0].set(ylabel="state x(t)")
        ax[0].get_xaxis().set_visible(False)
        ax[1].set(xlabel="time", ylabel="control a(t)")
        if show is True:
            plt.show()
        pass

    def plotODE(self):
        self.fp.plot(timeline=self.timeline, x_0=None)
        pass

    # FUNCTIONS
    def g(self, x):
        return (x - self.fp.x_0) ** 2

    def g_x(self, x):
        return 2 * (x - self.fp.x_0)

    def f(self, x, a):
        return self.fp.f(x) - a * x

    def L(self, x, a):
        raise NotImplemented

    def computeOptimalControl(self, x: np.array, y: np.array) -> np.array:
        raise NotImplemented

    def hamiltonian(self, x, y, a):  # not optimized hamiltonian
        return y * self.f(x, a) + self.L(x, a)

    def Hamiltonian(self, x, y):
        a = self.computeOptimalControl(x, y)
        return self.hamiltonian(x, y, a)

    def functional(self, x: np.array, a: np.array):
        J = self.g(x[-1])
        for i in range(self.N):
            J += self.L(x[i], a[i]) * self.dt + 1 / 2 * self.dt * (
                    self.L(x[i], a[i]) - self.L(x[i + 1], a[i + 1]))
        return J

    def H_x(self, x: np.array, y: np.array) -> np.array:
        a = self.computeOptimalControl(x, y)
        da_dx = (self.fp.c + self.fp.q * y)
        return y * self.fp.f_x(x) - a * da_dx

    def H_y(self, x: np.array, y: np.array) -> np.array:
        a = self.computeOptimalControl(x, y)
        da_dy = self.fp.q * x
        return self.fp.f(x) - a * da_dy

    pass


class QuadraticHamiltonian1D(_SingleSpeciesProblem):
    """
    f(x,a) = r X(K-X) - a X,
    L(x,a) = a^2/2 - a X
    g(x) = (X - X_0)^2 therefore g'(x) = 2 (X - X_0)

    Hamiltonian: H(x,y) = r y x(1-x/K) -1/2 (y+1)^2 x^2
    """

    def __init__(self, fishPool: SingleSpeciesPool = SingleSpeciesPool()):
        super().__init__(fishPool)

    def computeOptimalControl(self, x: np.array, y: np.array) -> np.array:
        a = (self.fp.c + self.fp.q * y) * x
        return a

    def L(self, x, a):
        return 1 / 2 * a ** 2 - self.fp.c * x * a

    pass


class LinearHamiltonian1D(_SingleSpeciesProblem):
    """
    L(x,a) = - c a X
    Hamiltonian: H(x,y) = -max{0, y+1} * alpha_max * x  + r y x(1-x/K)

    Note: this is not working due to the jumpdiscontinuity
    """

    def __init__(self, fishPool: SingleSpeciesPool = SingleSpeciesPool()):
        super().__init__(fishPool)
        pass

    def computeOptimalControl(self, x: np.array, y: np.array) -> np.array:
        a = self.fp.alpha_max * ((self.fp.c + self.fp.q * y) * x > 0)
        return a

    def L(self, x, a):
        return - self.fp.c * a * x

    pass


class RegularizedHamiltonian1D(LinearHamiltonian1D):
    """
    L(x,a) = - c a X
    Hamiltonian: H(x,y) = -max{0, (y+1)x} + alpha_max + r y x(1-x/K)
    H_x(x,y) = r y (1 - 2X/k) - (y + 1) alpha_max I(A) where A = {(y+1) x > 0}
    where we smooth out the heaviside (step) function as follows
    I(x>0) = 1 / (1 + exp(-2 / delta * x) for a particular k

    """
    delta = 100
    delta_min = 10 ** (-14)

    def __init__(self, fishPool: SingleSpeciesPool = SingleSpeciesPool()):
        super().__init__(fishPool)
        pass

    def s(self, x, y):  # regularization
        indicator = (self.fp.c + self.fp.q * y) * x
        return 1 / (1 + np.exp(-2 / self.delta * indicator))

    def computeOptimalControl(self, x: np.array, y: np.array) -> np.array:
        a = self.fp.alpha_max * self.s(x, y)
        return a

    def solveRecursively(self, fraction: float = 2, print_residual=False, print_delta=False):
        # initialization
        while self.delta > self.delta_min:
            if print_delta:
                print("delta:", self.delta)

            self.solve(print_residual=print_residual)
            self.solution.delta = self.delta
            self.delta = self.delta / fraction
        return self.solution

    pass


###
# Two Fish Species
###
class Solution2D:
    """ BLUEPRINT
        of Solution Class with different Trajectories which contains a state trajectory,
        x(t) = {x_t: t=0, ..., T }.
        And a costate trajectory,
        u(t) = {u_t: t=0, ..., T}.
        And control
        a = {a_t for t=0, ..., T}.
        """
    residual: float = None
    '''residual corresponding to the solution'''
    plot_name: str = None
    single_control: bool = False
    functional_value = None

    def __init__(self, x: Solution1D, y: Solution1D):
        self.timeline = x.timeline
        self.x: Solution1D = x
        ''' x is the trajectory of the first species '''
        self.y: Solution1D = y
        ''' y is the trajectory of the second species '''
        if x.plot_name is y.plot_name:
            self.plot_name = x.plot_name

        if np.array_equal(x.control, y.control):
            # print("this happened")
            self.single_control = True
        pass

    def addName(self, name):
        self.plot_name = name
        return self

    def passY(self):
        return np.append(self.x.passY(), self.y.passY())

    def plotTrajectory(self, plot=("density" or "harvesting" or "dual")):
        import matplotlib.pyplot as plt
        plt.Figure()
        if plot == "density":
            plt.plot(self.timeline, self.x.state)
            plt.plot(self.timeline, self.y.state)
        elif plot == "dual":
            plt.plot(self.timeline, self.x.costate)
            plt.plot(self.timeline, self.y.costate)
        elif plot == "harvesting":
            plt.plot(self.timeline, self.x.control)
            if not self.single_control:
                plt.plot(self.timeline, self.y.control)
        else:
            raise NameError("wrong input name")
        plt.xlabel("time (t)")
        plt.ylabel(plot + " (t)")
        plt.title("n =" + str(len(self.timeline) - 1) + "\niteration:" + self.plot_name)
        plt.show()
        pass

    def plotTrajectories(self, plot_show=True):
        import matplotlib.pyplot as plt

        plt.subplots(nrows=2, ncols=1)
        plt.suptitle("N =" + str(len(self.timeline) - 1) + ", u=" + str(np.round(self.functional_value, 5)))
        plt.subplot(2, 1, 1).plot(self.timeline, self.x.state, label="x(t)")
        plt.subplot(2, 1, 1).plot(self.timeline, self.y.state, label="y(t)")
        plt.ylabel("state x(t)")
        plt.gca().get_xaxis().set_visible(False)
        # plt.subplot(3, 1, 2).plot(self.timeline, self.x.costate)
        # plt.subplot(3, 1, 2).plot(self.timeline, self.y.costate)
        # plt.ylabel("costate y(t)")
        # plt.gca().get_xaxis().set_visible(False)
        plt.subplot(2, 1, 2).plot(self.timeline, self.x.control, label="a_1(t)")
        if not self.single_control:
            plt.subplot(2, 1, 2).plot(self.timeline, self.y.control, label="a_2(t)")
        plt.ylabel("control a(t)")
        plt.xlabel("time (t)")
        plt.subplot(2, 1, 1).legend(loc="upper left")
        plt.subplot(2, 1, 2).legend(loc="upper left")
        if plot_show:
            plt.show()
        pass

    def __mod__(self, other):
        return abs(self.residual - other.residual)

    def __sub__(self, other):
        return self.functional_value - self.functional_value

    def __le__(self, other):  # self <= other
        return self.functional_value <= other.functional_value

    pass


class _DoubleSpeciesProblem(_ControlProblem, ABC):
    """ A Fishing Problem is a combination of a Fish Pool (a set of parameters for the fish population)
        and a Control Problem (A collection of functions for solving the Fishing Problem)

        Fish harvesting problem with two species with ODES :
        x'(t) = r_1 x(t) (1 - 1/K_x (m_11 x(t) + m_12 y(t))) - q_1 x(t) a(t)
        y'(t) = r_2 y(t) (1 - 1/K_x (m_21 x(t) + m_22 y(t))) - q_1 x(t) a(t)

        Functional is variable but of the form:
        J(x,u) = int_0^T L(x,u) dt + g(x(t))

        L(x,u) = 'depends on situation'
        g(x(T)) = C (x(T) - x_0)^2

        With two controls a_x(t) and a_y(t) for each species
        """
    fp: DoubleSpeciesPool = None
    ''' set of parameters for the fish pool '''
    C = [1, 1]
    '''importance of terminal cost'''
    single_control: bool = False
    solution_dictionary: dict[int, Solution2D] = {}  # (iteration: int, Solution)

    def __init__(self, fishPool: DoubleSpeciesPool = DoubleSpeciesPool()):
        super().__init__()
        self.fp: DoubleSpeciesPool = fishPool
        pass

    # CALCULATE: SHORT TIME HORIZON PROBLEM
    def generateInitialCondition(self, version: str = "simple") -> np.array:
        """ Y = [x_0 ... x_0, 0 ... 0]  """
        x, u, y, v = np.split(np.zeros(4 * self.N), 4)

        if version == "simple":
            x, y = np.ones(self.N) * self.fp.x_0[0], np.ones(self.N) * self.fp.x_0[1]
            u, v = np.zeros(self.N), np.zeros(self.N)
        elif version == "calculated":
            x, y = self.fp.generateSolution(self.N, self.dt)
            u, v = self.generateCostates(self.N, x, y)

        return np.append(np.append(x, u), np.append(y, v))

    def generateCostates(self, dN: int, x: np.array, y: np.array):
        u, v = np.zeros_like(x), np.zeros_like(y)
        u[-1], v[-1] = self.gdot(x[-1], y[-1], x_0=None)
        for i in reversed(range(dN - 1)):
            u[i] = u[i + 1] + self.dt * self.H_x(x[i], u[i + 1], y[i], v[i + 1])
            v[i] = v[i + 1] + self.dt * self.H_y(x[i], u[i + 1], y[i], v[i + 1])
        return u, v

    def symplecticEulerF(self, Y: np.array) -> np.array:
        """
        (x,y)[n+1] = (x,y)[n] + dt H_(u,v)( (x,y)[n], (u,v)[n+1] )
        (u,v)[n] = (u,v)[n+1] + dt H_(x,y)( (x,y)[n], (u,v)[n+1] )
        """
        FY: np.array = np.zeros_like(Y)
        x, u, y, v = np.array_split(Y, 4)

        # FY[0] = -x[0] + self.fp.x_0 + self.stepSize  # * self.H_y(self.fp.x_0, u[1])
        # FY[0] = -x[0] + self.fp.x_0 + self.stepSize * self.H_y(self.fp.x_0, y[1])
        N = self.N

        FY[0] = -x[0] + self.fp.x_0[0] + self.dt * self.H_u(self.fp.x_0[0], u[0], self.fp.x_0[1], v[0])
        FY[2 * N] = -y[0] + self.fp.x_0[1] + self.dt * self.H_v(self.fp.x_0[0], u[0], self.fp.x_0[1], v[0])

        FY[1:N] = -x[1:N] + x[:N - 1] + self.dt * self.H_u(x[:N - 1], u[1:N], y[:N - 1], v[1:N])
        FY[N:2 * N - 1] = - u[:N - 1] + u[1:N] + self.dt * self.H_x(x[:N - 1], u[1:N], y[:N - 1], v[1:N])

        FY[1 + 2 * N:3 * N] = -y[1:N] + y[:N - 1] + self.dt * self.H_v(x[:N - 1], u[1:N], y[:N - 1], v[1:N])
        FY[3 * N:4 * N - 1] = - v[:N - 1] + v[1:N] + self.dt * self.H_y(x[:N - 1], u[1:N], y[:N - 1], v[1:N])

        FY[2 * N - 1] = -u[-1] + self.g_x(x[-1], None)
        FY[-1] = -v[-1] + self.g_y(y[-1], None)
        return FY

    def updateSolutionAndAddToDictionary(self, Y: np.array, residual: float, print_J=False):
        x, u, y, v = np.array_split(Y, 4)

        # First Species
        x = np.insert(x, 0, self.fp.x_0[0])
        initial_u = u[0] + self.dt * self.H_x(self.fp.x_0[0], u[0], self.fp.x_0[1], v[0])
        u = np.insert(u, 0, initial_u)
        # Second Species
        y = np.insert(y, 0, self.fp.x_0[1])
        initial_v = v[0] + self.dt * self.H_y(self.fp.x_0[0], u[0], self.fp.x_0[1], v[0])
        v = np.insert(v, 0, initial_v)

        if self.single_control:
            control = self.computeOptimalControl(x, u, y, v)
            x_solution: Solution1D = Solution1D(self.timeline, x, u, control)  # .addName(self.iteration)
            y_solution: Solution1D = Solution1D(self.timeline, y, v, control)  # .addName(self.iteration)
        else:
            control_x, control_y = self.computeOptimalControl(x, u, y, v)
            x_solution: Solution1D = Solution1D(self.timeline, x, u, control_x)  # .addName(self.iteration)
            y_solution: Solution1D = Solution1D(self.timeline, y, v, control_y)  # .addName(self.iteration)

        self.solution: Solution2D = Solution2D(x_solution, y_solution).addName(str(self.iteration))
        self.solution.residual = residual

        J = self.functional(x, u, y, v)
        self.solution.functional_value = J
        if print_J:
            print('J:', J)
        self.solution_dictionary[self.iteration] = self.solution
        pass

    def solve(self, print_residual=False, version: str = "simple", dN=None) -> np.array:
        """ F(Y) = 0 via fsolve where Y = (x,lambda_1,y,lambda_2)"""
        guess: np.array = np.zeros(self.N * 4)
        sol: np.array = np.zeros_like(guess)

        if not self.problemIsInitialized:
            if self.end > 1:
                print("Warning: time long horizon")
            guess = self.generateInitialCondition(version=version)
            self.problemIsInitialized = True
        elif self.problemHorizonIsAppended and dN is not None:
            guess = self.generateAppendedHorizonCondition(dN=dN)
        else:
            guess = self.solution.passY()

        if print_residual:
            t = time.time()

        sol = fsolve(func=self.symplecticEulerF, x0=guess)

        if print_residual:
            print(time.time() - t, "s")

        residual = sum(abs(self.symplecticEulerF(sol)))

        if print_residual:
            print("residual", residual)
        self.updateSolutionAndAddToDictionary(sol, residual)

        # if self.iteration == 0:
        #     self.solution.plotTrajectories()
        self.iteration += 1
        return sol

    # CALCULATE: LONG TIME HORIZON PROBLEM TODO
    def generateAppendedHorizonCondition(self, dN: int):
        x, u, y, v = np.split(self.solution.passY(), 4)

        x_add, y_add = self.fp.generateAppendedEnd(N=dN, timestep=self.dt, x_0=[x[-1], y[-1]])
        u_add, v_add = self.generateCostates(dN, x_add, y_add)

        x, y = np.append(x, x_add), np.append(y, y_add)
        u, v = np.append(u, u_add), np.append(v, v_add)

        return np.append(np.append(x, u), np.append(y, v))

    def solveAppendedProblem(self, dN):
        guess = self.generateAppendedHorizonCondition(dN)
        sol = fsolve(func=self.symplecticEulerF, x0=guess)

        residual = sum(abs(self.symplecticEulerF(sol)))

        print("residual", residual)
        self.updateSolutionAndAddToDictionary(sol, residual)
        return sol

    def solveLongTimeHorizon(self, T_0: float, T_end: float, N_max: int, N_0: int, dN: int):
        # dt = T_end/N_max
        # steps = (N_max - N_0) / dN
        N = N_0
        self.adjustSteps(N=N, end=T_0)
        self.solve(print_residual=False)
        k = 0

        while self.N < N_max:
            k += 1
            N += dN
            print("step:", N)
            self.appendHorizon(dN)
            self.solveAppendedProblem(dN)
            # self.solution.plotTrajectories()
            pass
        self.solution.plotTrajectories()
        pass

    # PLOTTING
    def plotODEs(self):
        self.fp.plotSolution(self.N, self.dt)
        pass

    def plotDictionary(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
        for item in self.solution_dictionary.items():
            key, value = item
            axs[0, 0].plot(value.timeline, value.x.state, label=key)
            axs[1, 0].plot(value.timeline, value.x.control, label=key)
            axs[0, 1].plot(value.timeline, value.y.state, label=key)
            axs[1, 1].plot(value.timeline, value.y.control, label=key)
            pass
        fig.suptitle("iterations:" + str(key) + " N:" + str(len(value.timeline) - 1))
        axs[0, 0].set(ylabel="state", title="Species 1")
        axs[0, 1].set(title="Species 2")

        min_lim, max_lim = min(min((axs[0, 1].get_ylim(), axs[0, 0].get_ylim()))), max(
            max(axs[0, 1].get_ylim(), axs[0, 0].get_ylim()))
        axs[0, 0].set_ylim(min_lim, max_lim)
        axs[0, 1].set_ylim(min_lim, max_lim)

        axs[0, 0].get_xaxis().set_visible(False)
        axs[0, 1].get_xaxis().set_visible(False)

        min_lim, max_lim = min(min((axs[1, 0].get_ylim(), axs[1, 1].get_ylim()))), max(
            max(axs[1, 0].get_ylim(), axs[1, 1].get_ylim()))
        axs[1, 0].set_ylim(min_lim, max_lim)
        axs[1, 1].set_ylim(min_lim, max_lim)
        axs[1, 0].set(ylabel="control", xlabel="time (t)")
        axs[1, 1].set(xlabel="time (t)")

        plt.show()

    def plotSolution(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
        key, value = self.iteration, self.solution

        axs[0, 0].plot(value.timeline, value.x.state, label=key)
        axs[1, 0].plot(value.timeline, value.x.control, label=key)
        axs[0, 1].plot(value.timeline, value.y.state, label=key)
        axs[1, 1].plot(value.timeline, value.y.control, label=key)

        fig.suptitle("iterations:" + str(key) + " N:" + str(len(value.timeline) - 1))
        axs[0, 0].set(ylabel="state", title="Species 1")
        axs[0, 1].set(title="Species 2")

        min_lim, max_lim = min(min((axs[0, 1].get_ylim(), axs[0, 0].get_ylim()))), max(
            max(axs[0, 1].get_ylim(), axs[0, 0].get_ylim()))
        axs[0, 0].set_ylim(min_lim, max_lim)
        axs[0, 1].set_ylim(min_lim, max_lim)

        axs[0, 0].get_xaxis().set_visible(False)
        axs[0, 1].get_xaxis().set_visible(False)

        min_lim, max_lim = min(min((axs[1, 0].get_ylim(), axs[1, 1].get_ylim()))), max(
            max(axs[1, 0].get_ylim(), axs[1, 1].get_ylim()))
        axs[1, 0].set_ylim(min_lim, max_lim)
        axs[1, 1].set_ylim(min_lim, max_lim)
        axs[1, 0].set(ylabel="control", xlabel="time (t)")
        axs[1, 1].set(xlabel="time (t)")

        plt.show()
        pass

    # FUNCTIONS
    def hamiltonian(self, x, y, a):  # not optimized hamiltonian
        # return y * self.f(x, a) + self.L(x, a)
        raise NotImplemented

    def L(self, x, u, y, v):
        raise NotImplemented

    def functional(self, x, u, y, v):
        J = self.g(x[-1], y[-1])

        for i in range(self.N):
            J += self.L(x[i], u[i], y[i], v[i]) * self.dt + 1 / 2 * self.dt * (
                    self.L(x[i], u[i], y[i], v[i]) - self.L(x[i + 1], u[i + 1], y[i + 1], v[i + 1]))
        return J

    def g(self, x, y):
        return self.C[0] * (x - self.fp.x_0[0]) ** 2 + self.C[1] * (y - self.fp.x_0[1]) ** 2

    def g_x(self, x, x_0=None):
        if x_0 is None:
            x_0 = self.fp.x_0[0]
        else:
            pass
        return 2 * self.C[0] * (x - x_0)  # FIXME

    def g_y(self, y, y_0=None):
        if y_0 is None:
            y_0 = self.fp.x_0[1]
        else:
            pass
        return 2 * self.C[1] * (y - y_0)  # FIXME

    def gdot(self, x, y, x_0: list = None):
        if x_0 is None:
            x_0 = self.fp.x_0
        else:
            pass
        return self.g_x(x, x_0[0]), self.g_y(y, x_0[1])

    def H_x(self, x, u, y, v):
        da_dx = self.fp.c[0] + u
        return u * self.fp.f_1_x(x, y) + v * self.fp.f_2_x(x, y) - self.a_x(x, u) * da_dx

    def H_y(self, x, u, y, v):
        da_dy = self.fp.c[1] + v
        return u * self.fp.f_1_y(x, y) + v * self.fp.f_2_y(x, y) - self.a_y(y, v) * da_dy

    def H_u(self, x, u, y, v):
        da_du = x
        return self.fp.f1(x, y) - self.a_x(x, u) * da_du

    def H_v(self, x, u, y, v):
        da_dv = y
        return self.fp.f2(x, y) - self.a_y(y, v) * da_dv

    def a_x(self, x, u):
        raise NotImplemented

    def a_y(self, y, v):
        raise NotImplemented

    def computeOptimalControl(self, x: float, u: float, y: float, v: float) -> tuple[float, float]:
        a_x = self.a_x(x, u)
        a_y = self.a_y(y, v)
        return a_x, a_y

    pass


class QuadraticHamiltonian2D(_DoubleSpeciesProblem):
    """
    L(x,y,a) = 1/2a^2 - c_x a_x x - c_y a_y y
    H(x,y) = L(x,y,a) + lambda * f(x,a)
    """

    def __init__(self, fishPool: DoubleSpeciesPool = DoubleSpeciesPool()):
        super().__init__(fishPool)
        pass

    def a_x(self, x, u):
        return (self.fp.c[0] + u) * x

    def a_y(self, y, v):
        return (self.fp.c[1] + v) * y

    def L(self, x, u, y, v):
        a = self.computeOptimalControl(x, u, y, v)
        reward_of_x = self.fp.c[0] * x * a[0]
        reward_of_y = self.fp.c[1] * y * a[1]
        return 1 / 2 * (a[0] ** 2 + a[1] ** 2) - (reward_of_x + reward_of_y)

    pass


class RegularizedHamiltonian2D(_DoubleSpeciesProblem):
    """
      L(x,y,a) = -c_1 a_x x - c_2 a_y y
      H(x,y) = u f_1 + v f_2 - a^T ( (c_1+q_1u) x, (c_2+q_2 v) y)^+
      """
    delta = [100, 100]
    delta_min = [10 ** (-10), 10 ** (-10)]

    def __init__(self, fishPool: DoubleSpeciesPool = DoubleSpeciesPool()):
        super().__init__(fishPool)
        pass

    # CALCULATING
    def solveRecursively(self, fraction: float = 2, print_residual=False, res=None, version="simple", print_J=False,solve_seperately=True):
        # initialization
        while any(self.delta[i] > self.delta_min[i] for i in range(2)):  # or self.delta[1] > self.delta_min[1]:
            if print_residual:
                print("delta:", self.delta)

            sol = self.solve(print_residual=print_residual, version="calculated")  # simple vs calculated
            if res is not None:
                print(abs(self.solution.residual - res))
            if solve_seperately:
                if self.delta[0] > self.delta_min[0] and self.delta[1] < self.delta_min[1]:
                    self.delta[0] = self.delta[0] / fraction
                elif self.delta[0] < self.delta_min[0] and self.delta[1] > self.delta_min[1]:
                    self.delta[1] = self.delta[1] / fraction
                else:
                    self.delta[self.iteration % 2] = self.delta[self.iteration % 2] / fraction
            else:
                self.delta[0] = self.delta[0] / fraction
                self.delta[1] = self.delta[1] / fraction
            pass
        # self.plotSolution()
        if print_J:
            print("j:", self.solution.functional_value)

        return sol

    # LONG HORIZON
    def solveAppendedProblem(self, dN, fraction=3):

        sol = self.solveRecursively(solve_seperately = False)

        residual = sum(abs(self.symplecticEulerF(sol)))

        print("residual", residual)
        self.updateSolutionAndAddToDictionary(sol, residual)
        self.delta = [100, 100]
        return sol
        #
        # while any(self.delta[i] > self.delta_min[i] for i in range(2)):
        #     # print("delta:", self.delta)
        #     sol = fsolve(func=self.symplecticEulerF, x0=guess)
        #     residual = sum(abs(self.symplecticEulerF(sol)))
        #     # print("residual", residual)
        #     self.updateSolutionAndAddToDictionary(sol, residual)
        #
        #     self.delta = self.delta / fraction
        # self.delta = [100,100]
        pass

    # def solveLongTimeHorizon(self,T_0: float,T_end:float, N_max:int, N_0: int, dN: int):
    #     # dt = T_end/N_max
    #     # steps = (N_max - N_0) / dN
    #     N = N_0
    #     self.adjustSteps(N=N, end=T_0)
    #     self.solve(print_residual=False)
    #     k = 0
    #
    #
    #     while self.N < N_max:
    #         k += 1
    #         N += dN
    #         print("step:", N)
    #         self.appendHorizon(dN)
    #         self.solveAppendedProblem(dN)
    #         self.solution.plotTrajectories()
    #         pass
    #     pass

    # FUNCTIONS
    def a_x(self, x, u):
        indicator = (self.fp.c[0] + u) * x
        s1 = 1 / (1 + np.exp(-2 / self.delta[0] * indicator))
        return self.fp.alpha_max[0] * s1

    def a_y(self, y, v):
        indicator = (self.fp.c[1] + v) * y
        s2 = 1 / (1 + np.exp(-2 / self.delta[1] * indicator))
        return self.fp.alpha_max[0] * s2

    def L(self, x, u, y, v):
        a = self.computeOptimalControl(x, u, y, v)
        reward_of_x = self.fp.c[0] * x * a[0]
        reward_of_y = self.fp.c[1] * y * a[1]
        return - (reward_of_x + reward_of_y)

    pass


class _DoubleSpeciesSingleControlProblem(_DoubleSpeciesProblem, ABC):
    """ A Fishing Problem is a combination of a Fish Pool (a set of parameters for the fish population)
        and a Control Problem (A collection of functions for solving the Fishing Problem)

        Fish harvesting problem with two species with ODES :
        x'(t) = r_1 x(t) (1 - m_11 x(t) - m_12 y(t)) - q_1 x(t) a(t)
        y'(t) = r_2 y(t) (1 - m_21 x(t) - m_22 y(t)) - q_1 x(t) a(t)

        Functional is variable but of the form:
        J(x,u) = int_0^T L(x,u) dt + g(x(t))

        L(x,u) = 'depends on situation'
        g(x(T)) = C (x(T) - x_0)^2

        With one control a(t), and only one maximal value namely alpha_max
        """
    single_control: bool = True
    alpha_max: float = 1

    def __init__(self, fishPool: DoubleSpeciesPool = DoubleSpeciesPool()):
        super().__init__(fishPool)
        self.fp: DoubleSpeciesPool = fishPool
        pass

    # CALCULATING
    def a(self, x, u, v, y):
        raise NotImplemented

    def updateSolutionAndAddToDictionary(self, Y: np.array, error: float, print_J=False):
        x, u, y, v = np.array_split(Y, 4)
        # First Species
        x = np.insert(x, 0, self.fp.x_0[0])
        initial_u = u[0] + self.dt * self.H_x(self.fp.x_0[0], u[0], self.fp.x_0[1], v[0])
        u = np.insert(u, 0, initial_u)
        # Second Species
        y = np.insert(y, 0, self.fp.x_0[1])
        initial_v = v[0] + self.dt * self.H_y(self.fp.x_0[0], u[0], self.fp.x_0[1], v[0])
        v = np.insert(v, 0, initial_v)

        control = self.a(x, u, y, v)
        x_solution: Solution1D = Solution1D(self.timeline, x, u, control)
        y_solution: Solution1D = Solution1D(self.timeline, y, v, control)

        self.solution: Solution2D = Solution2D(x_solution, y_solution).addName(str(self.iteration))
        self.solution.residual = error
        J = self.functional(x, u, y, v)
        self.solution.functional_value = J
        # self.solution.delta = self.delta
        # if print_J:
        #     print('J:', J)
        self.solution_dictionary[self.iteration] = self.solution
        pass

    # PLOTTING
    def plotSolution(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        ax1.plot(self.solution.timeline, self.solution.x.state, label=self.solution.plot_name)
        ax2.plot(self.solution.timeline, self.solution.y.state, label=self.solution.plot_name)
        ax3.plot(self.solution.timeline, self.solution.x.control, label=self.solution.plot_name)

        ax1.set(ylabel="state", title="Species 1")

        min_lim, max_lim = min(min((ax1.get_ylim(), ax2.get_ylim()))), max(
            max(ax1.get_ylim(), ax2.get_ylim()))
        ax1.set_ylim(min_lim, max_lim)
        ax2.set_ylim(min_lim, max_lim)
        ax2.set(title="Species 2")
        ax3.set(ylabel="control", xlabel="time")

        fig.show()

    def plotDictionary(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        for item in self.solution_dictionary.items():
            key, value = item
            ax1.plot(value.timeline, value.x.state, label=key)
            ax2.plot(value.timeline, value.y.state, label=key)
            ax3.plot(value.timeline, value.x.control, label=key)
            pass
        ax1.set(ylabel="state", title="Species 1")
        min_lim, max_lim = min(min((ax1.get_ylim(), ax2.get_ylim()))), max(
            max(ax1.get_ylim(), ax2.get_ylim()))
        ax1.set_ylim(min_lim, max_lim)
        ax2.set_ylim(min_lim, max_lim)
        ax2.set(title="Species 2")
        ax3.set(ylabel="control", xlabel="time")

        fig.show()
        pass

    def H_x(self, x, u, y, v):
        da_dx = self.fp.c[0] + u
        return u * self.fp.f_1_x(x, y) + v * self.fp.f_2_x(x, y) - self.a(x, u, y, v) * da_dx

    def H_y(self, x, u, y, v):
        da_dy = self.fp.c[1] + v
        return u * self.fp.f_1_y(x, y) + v * self.fp.f_2_y(x, y) - self.a(x, u, y, v) * da_dy

    def H_u(self, x, u, y, v):
        da_du = x
        return self.fp.f1(x, y) - self.a(x, u, y, v) * da_du

    def H_v(self, x, u, y, v):
        da_dv = y
        return self.fp.f2(x, y) - self.a(x, u, y, v) * da_dv

    pass


class QuadraticHamiltonian2DSingleControl(_DoubleSpeciesSingleControlProblem):
    """
    TODO
    """

    def __init__(self, fishPool: DoubleSpeciesPool = DoubleSpeciesPool()):
        super().__init__(fishPool)
        pass

    # FUNCTIONS
    def a(self, x, u, y, v) -> float:
        a_x = (self.fp.c[0] + u) * x
        a_y = (self.fp.c[1] + v) * y
        return a_x + a_y

    def L(self, x, u, y, v):
        a = self.a(x, u, y, v)
        reward_of_x = self.fp.c[0] * x * a
        reward_of_y = self.fp.c[1] * y * a
        return 1 / 2 * (a ** 2) - (reward_of_x + reward_of_y)

    pass


class RegularizedHamiltonian2DSingleControl(_DoubleSpeciesSingleControlProblem):  # TODO: FIX THIS
    """
      TODO
    """
    delta = 100
    delta_min = 10 ** (-10)

    def __init__(self, fishPool: DoubleSpeciesPool = DoubleSpeciesPool()):
        super().__init__(fishPool)
        pass

    def solveAppendedProblem(self, dN):
        # guess = self.generateAppendedHorizonCondition(dN)
        sol = self.solveRecursively()

        residual = sum(abs(self.symplecticEulerF(sol)))

        print("residual", residual)
        self.updateSolutionAndAddToDictionary(sol, residual)
        return sol

    def solveRecursively(self, fraction: float = 2, print_residual=False, res=None, guess=None, version="simple"):
        # initialization
        while self.delta > self.delta_min:
            if print_residual:
                print("delta:", self.delta)

            sol = self.solve(print_residual=print_residual, version="calculated")  # simple vs calculated
            if res is not None:
                print(abs(self.solution.residual - res))
            self.delta = self.delta / fraction
        self.delta = 100
        return sol

    # FUNCTIONS

    def a(self, x, u, y, v) -> float:
        indicator = (self.fp.c[0] + u) * x + (self.fp.c[1] + v) * y
        s = 1 / (1 + np.exp(-2 / self.delta * indicator))
        return self.alpha_max * s

    def L(self, x, u, y, v):
        a = self.a(x, u, y, v)
        reward_of_x = self.fp.c[0] * x * a
        reward_of_y = self.fp.c[1] * y * a
        return - (reward_of_x + reward_of_y)

    pass


###
### OLD Classes
###
class _HarvestingForProfit1D(LinearHamiltonian1D):
    #     """
    #     f(x, a) = r X(1-X/K) - q a X, where q is catchability
    #     h(x,a) = p q x(t) a(t) - c a(t)
    #     g(x) = 0 (!)
    #
    #     Hamiltonian: H(x,y) = pq x(t) a(t) - c a(t) + y f(x,a)
    #
    #     ## NOTE: We let q only be used in the final computation of a!
    #
    #     u == 0 if q x(t) (p  - y(t)) c < 0
    #     u == E if q x(t) (p  - y(t)) c > 0
    #
    #     H_x(x,y) = -( p q a(t) +(r(1 - 2/K x(t)) - qa(t)) y)
    #     H_y(x,y) = r x(1-x/K) - qxE (x(t) != 0)
    #     """
    #
    #     def __init__(self, fishPool: SingleSpeciesPool = SingleSpeciesPool()):
    #         super().__init__(fishPool)
    #         pass
    #
    #     def g(self, x):
    #         g = np.zeros_like(x)
    #         return g
    #
    #     def g_x(self, x):
    #         g = np.zeros_like(x)
    #         return g
    #
    #     def L(self, x, a):
    #         return self.fp.ppf * a * x - a * self.fp.unit_cost / self.fp.q
    #
    #     def Hamiltonian(self, x, y):
    #         indicator = (x * (self.fp.ppf - y) - self.fp.unit_cost / self.fp.q > 0)
    #         return self.hamiltonian(x, y, indicator)
    #
    #     def hamiltonian(self, x, y, a):
    #         return a * (x * (self.fp.ppf - y) - self.fp.unit_cost / self.fp.q) \
    #                + y * self.fp.r * x * (1 - x / self.fp.K)
    #
    #     def H_y(self, x, y):
    #         indicator = (x * (self.fp.ppf - y) - self.fp.unit_cost / self.fp.q > 0)
    #         a = self.fp.alpha_max * indicator
    #         print(a)
    #         return -(self.fp.ppf * a + y * (self.fp.r * (1 - 2 / self.fp.K * x) - a))
    #
    #     def H_x(self, x, y):
    #         indicator = (x * (self.fp.ppf - y) - self.fp.unit_cost / self.fp.q > 0)
    #         a = self.fp.alpha_max * indicator
    #         return self.f(x, a)
    #
    #     def computeControlArray(self, x_array: np.array, y_array: np.array) -> np.array:
    #         xdot = self.H_y(x_array, y_array)
    #         a_array = self.fp.r * (1 - x_array / self.fp.K) - (1 / x_array) * xdot
    #         return a_array / self.fp.q
    pass


class _ProportionalHarvesting1D(_SingleSpeciesProblem):
    #     """
    #     f(x, a) = r X(1-X/K) - a X / (1+X),
    #     h(x,a) = - a X / (1+ s_0 X) + a^2/2
    #     g(x) = (X - X_0)^2 therefore g'(x) = 2 (X - X_0)
    #
    #     Hamiltonian: H(x,y) = r y x(1-x/K) -1/2 (y+1)^2 (x/(1+X)^2
    #     """
    #
    #     def __init__(self, fishPool: SingleSpeciesPool = SingleSpeciesPool()):
    #         super().__init__(fishPool)
    #         pass
    #
    #     @staticmethod
    #     def runningCost(x, a):
    #         return - a * x / (1 + x) + a ** 2 / 2
    #
    #     def Hamiltonian(self, x, y):
    #         a = (1 + y) * x / (1 + self.fp.s_0 * x)
    #         return self.hamiltonian(x, y, a)
    #
    #     def H_x(self, x, y):
    #         return self.fp.r * y * (1 - 2 * x / self.fp.K) - \
    #                (1 + y) ** 2 * (2 * x - 2 * (self.fp.s_0 - 1) * x ** 2) / (1 + self.fp.s_0 * x) ** 3
    #
    #     def H_y(self, x, y):
    #         return self.fp.r * x * (1 - x / self.fp.K) - \
    #                (1 + y) * (x / (1 + self.fp.s_0 * x)) ** 2
    pass
