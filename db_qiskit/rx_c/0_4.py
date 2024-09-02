from qiskit import QuantumCircuit
from math import pi
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        qc.h(Mod(i0, n))
    for i0 in range(n):
        for i1 in range(abs(n + i0 + 1)):
            qc.rx(2 ** (-i1 + 2 * n) * pi, Mod(i1 - 5, n))
    for i0 in range(n):
        qc.x(Mod(2, n))
    for i0 in range(n):
        qc.h(Mod(i0 + 3, n))
    qc.rx(pi / 2 ** n, Mod(1, n))
    for i0 in range(n):
        qc.x(Mod(i0 - 4, n))
    for i0 in range(n):
        qc.rx(pi / (2 ** (i0 - n) + 2 * n + 1), Mod(-1, n))
    for i0 in range(n):
        for i1 in range(abs(i0 - n - 0)):
            qc.rx(pi / (2 ** i0 + i1 + n + 1), Mod(i0 + 1, n))
    for i0 in range(n):
        for i1 in range(abs(n + i0 - 1)):
            qc.ry(pi / (i1 - 2 * n + 1), Mod(i0 - 2, n))
    return qc