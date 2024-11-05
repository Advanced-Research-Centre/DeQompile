from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        qc.h(int(0))
    for i0 in range(n):
        qc.rz(pi / (2 ** (3 * n) + i0), int(Mod(i0 + 6, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 + n + 1)):
            qc.ry(pi / (2 ** (i0 + n) + 1), int(Mod(i0 - 3, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 - i0 + 0)):
            qc.rx(pi / (2 ** (i0 + n) + i1), int(Mod(i0 + 1, n)))
    for i0 in range(n):
        for i1 in range(abs(n + i0 + 1)):
            qc.ry(pi / (2 ** (3 * n) + i1 + n), int(Mod(2, n)))
    return qc