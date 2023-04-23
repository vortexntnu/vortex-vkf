#!/usr/bin/python
from casadi import *
import numpy as np


def symbolic_VL(A, D, G, n):
    Ts = SX.sym("Ts")

    UL = -A
    UR = G @ D @ G.T
    LL = SX.zeros((n, n))
    LR = A.T

    exponent = blockcat(UL, UR, LL, LR)
    #exponent = MX.sym([[      -A,         G @ D @ G.T],
    #[SX.zeros((n, n)), A.T]])

    #exponent = MX.sym()
    VL = expm(exponent * Ts)

    V1 = VL[n:2 * n, n:2 * n]
    V2 = VL[0:n, n:2 * n]

    Q = V1.T @ V2

    print(Q)


def main():
    n = 4

    A = SX.zeros((n, n))

    G = SX.eye(n)
    D = SX.sym('s', Sparsity.diag(n))

    symbolic_VL(A, D, G, n)


if __name__ == '__main__':
    main()
