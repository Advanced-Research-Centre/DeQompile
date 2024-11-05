from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        for i1 in range(abs(n + i0 - 1)):
            qc.rx(pi / (2 ** (i0 + n) + i0 - n + 1), int(Mod(-6, n)))
    for i0 in range(n):
        qc.rz(pi / (i0 - 2 * n + 1), int(Mod(i0 - 1, n)))
    for i0 in range(n):
        qc.rz(pi / (2 ** i0 + i0 - n + 1), int(Mod(i0 + 4, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 + i0 - 0)):
            qc.h(int(0))
    for i0 in range(n):
        for i1 in range(abs(n - n - 0)):
            qc.ry(pi / (2 ** (i0 + n) + n), int(Mod(1, n)))
    for i0 in range(n):
        qc.x(int(0))
    for i0 in range(n):
        qc.h(int(Mod(i0 + 1, n)))
    for i0 in range(n):
        qc.rx(pi, int(0))
    return qc