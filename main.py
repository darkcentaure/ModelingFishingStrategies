# Python Packages
# import time
# import numpy as np

# User-defined Objects
import time

import FishingPools
from FishingPools import *
from OptimalControl import *
from plotting import *

if __name__ == '__main__':
    # plotRampFunction()
    plotIndicatorFunction(True)

    exit(4)
    singleVsDoubleControlExperiment57()
    exit(4)
    fp = FishingPools.DoubleSpeciesPool().adjustParameters(
        r = [1,1],
        K = [1,0.75],
        M = [[1,0.3],[0.6,1]],
        x_0=[.5,.55],
        alpha_max = [1,1],
        q = [1,1],
        c = [1,1]
    )
    fp.plotSolution(100, 0.1)
    fish = QuadraticHamiltonian2D(fp).adjustSteps(200,10)
    fish.solve()
    fish.solution.plotTrajectories()

    fish = RegularizedHamiltonian2D(fp).adjustSteps(200,10)
    fish.solveRecursively()
    fish.solution.plotTrajectories()


    exit(3)


    plt.Figure()
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    fp.K = [100, 100]
    j = 0
    # fp.M[0][1] = 0.1/ fp.K[0]
    # fp.M[1][0] = 0.2/fp.K[1]
    for x_0 in [i*10 for i in range(1,11)]:
        fp.x_0[0] = x_0
        for x in [i * 10 for i in range(1,11)]:
            fp.x_0[1] = x
            x, y = fp.generateSolution(200, 0.05)

            plt.plot(x, y, c="C0", lw = 0.5)
            plt.arrow(x[25], y[25], x[26]-x[24], y[26]-y[24], shape='full', lw=0, length_includes_head=True, head_width=2)
            j += 1
            print(j)
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    # plt.legend(loc="right")
    plt.show()
    pass

    exit(3)


    fish = nSpeciesPool(2)
    fish.plotSolution(200,0.01)
    exit()

    # fish = RegularizedHamiltonian2DSingleControl().adjustSteps(400,8)
    # fish.plotODEs()
    # fish.C = [8, 8]
    # fish.()
    # fish.solution.plotTrajectories()
    fish = RegularizedHamiltonian2D().adjustSteps(200, 8)
    fish.C = [1, 1]
    # fish.solveRecursively()
    fish.solveLongTimeHorizon(2, 8, 200, 50, 10)
    fish.solution.plotTrajectories()
    exit(4)
    fish = FishingPools.SingleSpeciesPool()
    t = generateTimeline(0,6,200)
    fish.plot(t,[i*0.10 for i in range(10)])
    exit(4)
    fish = RegularizedHamiltonian2D().adjustSteps(N = 300, end = 10)
    fish.solveRecursively()
    fish.C = [10,10]
    fish.solution.plotTrajectories()
    exit(3)
    # fish = QuadraticHamiltonian2D().adjustSteps(300,10)
    # fish.plotODEs()
    # fish.C = [10,10]
    # fish.solve()
    # fish.solution.plotTrajectories()
    # exit(4)

    exit(4)
    fish = RegularizedHamiltonian1D().adjustSteps(200, 4)
    fish.plotODE()
    fish.C = 1
    # fish.solve()
    fish.solveRecursively()
    print(fish.solution.functional_value)
    fish.solution.plotTrajectories()
    exit(3)

    fish = QuadraticHamiltonian2D()
    fish.solve()
    fish = DoubleSpeciesPool()
    print(fish.M[0][1])
    fish.plotDifference(20, 1, [0.5, 0.45])
    exit(4)
    fish = RegularizedHamiltonian2D().adjustSteps(200, 10)
    fish.plotODEs()
    fish.C = [10,10]
    fish.solveRecursively(print_residual=False)
    print(fish.delta)
    exit(3)
    print("j:", fish.solution.functional_value)
    fish = RegularizedHamiltonian2D()
    fish.solveLongTimeHorizon(T_0=2, T_end=10, N_max=250, N_0=50, dN=10)
    print("j:", fish.solution.functional_value)
    fish = RegularizedHamiltonian2D().adjustSteps(250, 10)
    fish.solve()
    print("j:", fish.solution.functional_value)
    pass
