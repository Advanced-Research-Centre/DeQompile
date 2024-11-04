from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        qc.x(int(Mod(i0 - 1, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 - n + 1)):
            qc.ry(pi / (2 ** (i0 + n) + i1 - n), int(Mod(i1 - 1, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 + i0 - 1)):
            qc.rx(pi / (2 ** i0 + 2 * n + 1), int(Mod(i1 - 1, n)))
    return qc