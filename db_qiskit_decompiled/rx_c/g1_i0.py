from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        for i1 in range(abs(n - i0 - 1)):
            qc.rz(pi / (2 ** i1 + i1 - n + 2), int(Mod(i1 + 1, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 + i0 - 0)):
            qc.rz(pi / (2 ** (i0 - n) + i1 + 2 * n), int(Mod(i1 + 6, n)))
    for i0 in range(n):
        for i1 in range(abs(n - n + 0)):
            qc.rz(2 ** n * pi / (2 ** n * n + 1), int(Mod(i1 - 3, n)))
    for i0 in range(n):
        qc.rx(pi / (2 ** (2 * n) + 1), int(Mod(-3, n)))
    for i0 in range(n):
        qc.x(int(Mod(i0, n)))
    for i0 in range(n):
        qc.rx(pi / (2 ** (3 * n) + i0 - n), int(Mod(i0 - 3, n)))
    for i0 in range(n):
        qc.ry(pi / (2 ** (i0 - n) + i0 + n), int(Mod(3, n)))
    for i0 in range(n):
        qc.x(int(Mod(i0 + 3, n)))
    for i0 in range(n):
        for i1 in range(abs(n + n + 1)):
            qc.rx(pi / (2 ** (i0 + n) + i1), int(Mod(-2, n)))
    for i0 in range(n):
        qc.rx(pi / (2 ** n + i0 - n + 1), int(Mod(4, n)))
    return qc