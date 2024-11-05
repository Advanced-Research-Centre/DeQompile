from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        qc.x(int(Mod(i0, n)))
    for i0 in range(n):
        qc.h(int(Mod(i0 - 1, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 + i0 - 1)):
            qc.h(int(Mod(1, n)))
    for i0 in range(n):
        for i1 in range(abs(n - n + 0)):
            qc.h(int(Mod(i0 - 5, n)))
    for i0 in range(n):
        for i1 in range(abs(n - n - 1)):
            qc.h(int(Mod(i1, n)))
    for i0 in range(n):
        qc.rx(pi / (2 ** n + 2), int(Mod(3, n)))
    for i0 in range(n):
        qc.rz(pi / (2 ** n + i0 + n), int(0))
    for i0 in range(n):
        for i1 in range(abs(i0 + i0 + 0)):
            qc.rz(pi / (2 ** (i1 - 3 * n) + i1 - n), int(Mod(i1 + 4, n)))
    for i0 in range(n):
        qc.ry(pi / (2 ** (3 * n) + i0 - n), int(Mod(i0, n)))
    return qc