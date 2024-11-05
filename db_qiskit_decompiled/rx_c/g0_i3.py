from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        qc.x(int(Mod(i0 + 1, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 + i0 + 1)):
            qc.rx(pi / (2 ** (i1 + n) + 1), int(Mod(-1, n)))
    for i0 in range(n):
        qc.ry(pi / 4 ** n, int(0))
    for i0 in range(n):
        for i1 in range(abs(n - i0 - 0)):
            qc.ry(pi / (2 ** n + i1), int(Mod(4, n)))
    for i0 in range(n):
        qc.x(int(Mod(i0 + 3, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 - n + 1)):
            qc.x(int(Mod(-1, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 + n + 0)):
            qc.rx(pi / (2 ** n + i1 + n + 1), int(Mod(2, n)))
    return qc