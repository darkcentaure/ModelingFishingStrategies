"""
User-defined Python classes for Optimal Fishing Strategy Objects
and for Optimal Control Objects

Author: Niels in 't Veld
Created: 19/02/2022
Updated: 27/04/2022
Version: 4.0
"""
import random

import numpy as np
from scipy.optimize import fsolve


###
# Blueprints
###
class _FishPool:
    """
    FishPool is a blueprint for the user defined object fish pool,
    which is a set of parameters, corresponding to the differential equation
    x' = f(x),
    where x is the density of fish(es).
    A plot function, plotting the evolution of the fish density.
    """
###
# n Fish Species
###
class nSpeciesPool(_FishPool):
    def __init__(self, n):
        self.n = n
        self.K = np.ones(n)
        self.r = np.ones(n)
        self.M = self.generateRandomM(n)
        #np.fill_diagonal(M, 1)
        self.x_0 = 1/2 * np.ones(n)
        self.alpha_max = np.ones(n)
        self.c = np.ones(n)
        pass

    def generateRandomM(self, n):
        random.seed(4)
        M = np.random.rand(n, n)
        np.fill_diagonal(M, 1)
        return M

    def adjustParameters(self, r, K, M, x_0, alpha_max, q, c):
        self.r = r
        self.r = K
        self.x_0 = x_0
        self.M = M
        self.alpha_max = alpha_max
        self.c = c
        return self

    def f(self, x: np.array, i:int):
        return self.r[i] * x[i] * (1 - 1 / self.K[i] * (self.M[i][:].dot(x)))

    def fdot(self, x: np.array, i:int, j:int):
        """derivative of f_i(x) wrt x_j"""
        if i == j:
            return self.r[i] * (1 - 1 / self.K[i] * (x[i] + self.M[i][:].dot(x)))
        else:
            return - self.r[i] * x[i] / self.K[i] * self.M[i][j]
        pass

    def F(self, Y: np.array, args) -> np.array:
        x_0: np.array = args[0]
        timestep: float = args[1]

        N = len(Y)/self.n

        FY = np.zeros_like(Y)

        for k in range(self.n):
            # k is number of species
            l = (0 + i * N for i in range(self.n))
            FY[k*N] = -Y[k*N] + x_0[k] + timestep * self.f(Y[l], k)

        for s in range(1,N+1):
            for k in range(self.n):
                l = (s + i * N for i in range(self.n))
                FY[s+k*N] = -Y[s+k*N] + Y[s - 1 + k * N] + timestep * self.f(Y[l],k)

        return FY

    def generateAppendedEnd(self, N: int, timestep: float, x_0: list):
        sol, _ = self.solveViaBackwardEuler(N, timestep, x_0)
        return np.split(sol, self.n)

    def solveViaBackwardEuler(self, N: int, timestep: float, x_0: list) -> tuple[np.array, np.array]:
        if x_0 is None:
            x_0 = self.x_0
        guess = self.generateForwardEulerEvolution(N, timestep, x_0)

        args = [x_0, timestep]

        sol = fsolve(func=self.F, x0=guess, args=args)
        return sol, guess

    def forwardEuler(self, x_in, timestep):
        x_out = np.zeros(self.n)
        for k in range(self.n):
            x_out[k] = x_in[k] + timestep * self.f(x_in, k)
        return x_out

    def generateForwardEulerEvolution(self, N, timestep, x_0):
        # global scheme
        FY = np.zeros(N * 2)
        l = [0 + i * N for i in range(self.n)]
        if x_0 is None:
            FY[l] = self.x_0
        else:
            FY[l] = x_0

        for s in range(1,N+1):
            for k in range(self.n):
                l = (s + i * N for i in range(self.n))
                lplus1 = (s + 1 + i * N for i in range(self.n))
                FY[lplus1] = self.forwardEuler(FY[l], timestep)
        return FY

    def generateSolution(self, N: int, dt: float) -> tuple[np.array, np.array]:  # (x,y)
        Y = self.generateAppendedEnd(N, dt, self.x_0)
        return Y

    def plotSolution(self, N, dt):
        timeline = np.array([i * dt for i in range(N)])

        Y = self.generateSolution(N, dt)

        N = len(Y)/self.n

        import matplotlib.pyplot as plt
        plt.Figure()
        plt.xlabel("t")
        plt.ylabel("Density")
        for k in range(self.n):
            plt.plot(timeline, Y[k*N: (k+1)*N], label=k)
        plt.legend(loc="right")
        plt.show()
        pass

###
# Two Fish Species
###
class DoubleSpeciesPool(_FishPool):
    K = [1, 1]
    ''' max sustainable population'''
    r = [1/2, 1]
    '''growth_rate r = [r1,r2]'''
    M = [[1, 3], [3, 1]]
    '''matrix of inter-species-parameters
    # M = [[M_11, M_12], [M_21, M_22]
    # where M_11: , M_22:
    # M_12: , M_21:'''
    x_0 = [0.5, 0.6]
    '''initial population #x_0 = [x_0, y_0]'''
    alpha_max = [1, 1]
    '''max harvesting capacity'''
    q = [1, 1]
    '''catchability #q = [q_1, q_2]'''
    c = [1, 1]
    '''price per fish c = [c_1, c_2]'''

    def __init__(self):
        super().__init__()

    def adjustParameters(self, r, K, M, x_0, alpha_max, q, c):
        self.r = r
        self.r = K
        self.x_0 = x_0
        self.M = M
        self.alpha_max = alpha_max
        self.q = q
        self.c = c
        return self

    def f1(self, x, y):
        return self.r[0] * x * (1 - 1 / self.K[0] * (x + self.M[0][1] * y))

    def f_1_x(self, x, y):
        return self.r[0] * (1 - 1 / self.K[0] * (2 * x + self.M[0][1] * y))

    def f_1_y(self, x, y):
        return - self.r[0] * self.M[0][1] * x / self.K[0]

    def f2(self, x, y):
        return self.r[1] * y * (1 - 1 / self.K[1] * (self.M[1][0] * x + y))

    def f_2_x(self, x, y):
        return - self.r[1] * self.M[1][0] * y / self.K[1]

    def f_2_y(self, x, y):
        return self.r[1] * (1 - 1 / self.K[1] * (self.M[1][0] * x + 2 * y))

    def F(self, Y: np.array, args) -> np.array:
        x_0: list = args[0]
        timestep: float = args[1]

        x, y = np.split(Y, 2)
        N = len(x)

        FY = np.zeros_like(Y)

        FY[0] = -x[0] + x_0[0] + timestep * self.f1(x[0], y[0])
        FY[N] = -y[0] + x_0[1] + timestep * self.f2(x[0], y[0])

        FY[1:N] = - x[1:] + x[0:N - 1] + timestep * self.f1(x[1:], y[1:])
        FY[N + 1:] = - y[1:] + y[0:N - 1] + timestep * self.f2(x[1:], y[1:])
        return FY

    def generateAppendedEnd(self, N: int, timestep: float, x_0: list):
        sol, _ = self.solveViaBackwardEuler(N, timestep, x_0)
        x, y = np.split(sol, 2)
        return x, y

    def solveViaBackwardEuler(self, N: int, timestep: float, x_0: list) -> tuple[np.array, np.array]:
        if x_0 is None:
            x_0 = self.x_0
        guess = self.generateForwardEulerEvolution(N, timestep, x_0)

        args = [x_0, timestep]

        sol = fsolve(func=self.F, x0=guess, args=args)
        return sol, guess

    def forwardEuler(self, x_in, y_in, timestep):
        x_out = x_in + timestep * self.f1(x_in, y_in)
        y_out = y_in + timestep * self.f2(x_in, y_in)
        return x_out, y_out

    def generateForwardEulerEvolution(self, N, timestep, x_0):
        # global scheme
        FY = np.zeros(N * 2)

        if x_0 is None:
            FY[0] = self.x_0[0]
            FY[N] = self.x_0[1]
        else:
            FY[0] = x_0[0]
            FY[N] = x_0[1]

        for i in range(N - 1):
            FY[i + 1], FY[i + N + 1] = self.forwardEuler(FY[i], FY[N + i], timestep)
        # x,y = np.split(FY,2)
        return FY

    def plotDifference(self, N: int, dt: float, x_0: list):
        if x_0 is None:
            x_0 = self.x_0
        sol, guess = self.solveViaBackwardEuler(N, dt, x_0)
        x, y = np.split(sol, 2)
        x_guess, y_guess = np.split(guess, 2)

        timeline = [(i + 1) * dt for i in range(N)]

        import matplotlib.pyplot as plt
        plt.Figure()
        plt.xlabel("t")
        plt.ylabel("Density (t)")
        plt.plot(timeline, np.insert(x[:-1], 0, x_0[0]), c='darkred', label='x_backward')
        plt.plot(timeline, x_guess, c='red', label='x_forward')
        plt.plot(timeline, np.insert(y[:-1], 0, x_0[1]), c='darkblue', label='y_backward')
        plt.plot(timeline, y_guess, c='blue', label='y_forward')
        plt.legend(loc="lower right")
        plt.show()
        pass

    def generateSolution(self, N: int, dt: float) -> tuple[np.array, np.array]:  # (x,y)
        x, y = self.generateAppendedEnd(N, dt, self.x_0)

        return x, y

    def plotPhasePlot(self, N , dt):
        # timeline = np.array([i * dt for i in range(N)])

        x, y = self.generateSolution(N, dt)
        import matplotlib.pyplot as plt
        plt.Figure()
        plt.xlabel("t")
        plt.ylabel("Density")
        plt.plot(x, y, label="x")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.legend(loc="right")
        plt.show()
        pass

    def plotSolution(self, N, dt):
        timeline = np.array([i * dt for i in range(N)])

        x, y = self.generateSolution(N, dt)

        import matplotlib.pyplot as plt
        plt.Figure()
        plt.xlabel("t")
        plt.ylabel("Density")
        plt.plot(timeline, x, label="x")
        plt.plot(timeline, y, label="y")
        plt.legend(loc="right")
        plt.show()
        pass

    def __str__(self):
        string = "2D-Fish pool r:" + str(self.r) + ", etc."
        return string

    pass


###
# One Fish Species
###
class SingleSpeciesPool(_FishPool):
    """
    Single Species Fish Pool

    The fish follow the logistic equation, which is defined as follows
    x'(t) = r x(1-x/K),
    where r is the growth rate, or the relative growth rate coefficient;
    and K is the carrying Capacity, the carrying capacity of the population;
    defined by ecologists as the maximum population size that a particular environment can sustain.

    The differential equation is separable, therefore, it is analytic and
    x(t) = K / (1 + A exp(-rt)) with A = (K-P_0)/P_0,
    where P_0 is the initial population.
    """
    r = 1
    '''relative growth rate coefficient'''
    K = 0.75
    '''max population'''
    x_0 = .50
    '''initial population'''
    alpha_max = 1
    # '''max harvesting capacity'''
    # q = 1
    '''catchability'''
    ppf = 1
    '''price per fish'''
    c = 1

    def __init__(self):
        super().__init__()

    def adjustParameters(self, r, K, x_0, alpha_max, q, c):
        self.r = r
        self.K = K
        self.x_0 = x_0
        self.alpha_max = alpha_max
        self.q = q
        self.c = c
        return self

    def f(self, x):
        return self.r * x * (1 - x / self.K)

    def f_x(self, x):
        return self.r * (1 - 2 * x / self.K)

    def initialSolution(self, t: np.array) -> np.array:
        A = (self.K - self.x_0) / self.x_0
        return self.K / (1 + A * np.exp(-self.r * t))

    def solutionForDifferentInitialCondition(self, t: np.array, x_0: float) -> np.array:
        if x_0 == 0:
            return np.zeros_like(t)
        else:
            A = (self.K - x_0) / x_0
            return self.K / (1 + A * np.exp(-self.r * t))

    def plot(self, timeline: np.array, x_0, plot_eqs = False):
        fx = None
        if type(x_0) is list:
            pass
        elif x_0 is None:
            fx: np.array = self.initialSolution(timeline)
        elif type(x_0) is float or int:
            fx: np.array = self.solutionForDifferentInitialCondition(timeline, x_0)
        else:
            # type(x_0) is list
            raise NotImplemented("this should not happen")
        import matplotlib.pyplot as plt
        plt.Figure()
        if fx is None:
            for i in x_0:
                fx: np.array = self.solutionForDifferentInitialCondition(timeline, i)
                plt.plot(timeline, fx, c='C0')
        else:
            plt.plot(timeline, fx)
        if plot_eqs:
            plt.plot(timeline, self.K * np.ones_like(timeline), c='C1')
            plt.plot(timeline, np.zeros_like(timeline), c='C1')
        plt.xlabel("time (t)")
        plt.ylabel("x(t)")
        plt.title("Evolution of Fish Population")
        plt.show()
        pass

    # def calculateEquilibriumSolution(self):
    #     def

    def __str__(self):
        string = "1D-Fish pool r:" + str(self.r) + ", etc."
        return string

    pass
