from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        for i1 in range(abs(n - n + 1)):
            qc.ry(pi / (2 ** (2 * n) + 2 * n + 3), int(Mod(i0 - 2, n)))
    for i0 in range(n):
        qc.ry(pi / (2 ** (i0 - 2 * n) + i0 + n), int(Mod(i0 - 2, n)))
    for i0 in range(n):
        for i1 in range(abs(n - n + 1)):
            qc.ry(pi / (2 ** (i0 + n) + i1 + 2 * n + 1), int(Mod(i0 + 4, n)))
    for i0 in range(n):
        for i1 in range(abs(n + i0 - 0)):
            qc.rx(pi / (2 ** (i0 - n) + 1), int(Mod(i1 - 6, n)))
    for i0 in range(n):
        qc.h(int(Mod(i0 + 6, n)))
    for i0 in range(n):
        qc.rz(pi / (2 ** i0 + n + 1), int(Mod(3, n)))
    for i0 in range(n):
        for i1 in range(abs(n + i0 + 1)):
            qc.ry(pi / (2 ** n + i0 + n), int(Mod(i1, n)))
    return qc