from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        for i1 in range(abs(n - i0 - 0)):
            qc.rz(pi / (2 ** (i1 + 2 * n) + i1 - n), int(Mod(i0 - 1, n)))
    for i0 in range(n):
        qc.ry(pi / (2 ** (i0 + 2 * n) + 2 * n), int(0))
    for i0 in range(n):
        for i1 in range(abs(i0 - n - 1)):
            qc.ry(pi / (2 ** (i0 + n) + i0 + n), int(Mod(i1 + 3, n)))
    return qc