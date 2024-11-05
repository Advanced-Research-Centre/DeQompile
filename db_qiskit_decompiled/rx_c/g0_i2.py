from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        qc.ry(pi / (2 ** n + i0 + n), int(0))
    for i0 in range(n):
        for i1 in range(abs(i0 + i0 + 1)):
            qc.rz(pi / (i0 + n + 1), int(Mod(i1 - 1, n)))
    for i0 in range(n):
        qc.h(int(Mod(i0 - 2, n)))
    for i0 in range(n):
        qc.x(int(Mod(-5, n)))
    for i0 in range(n):
        qc.x(int(Mod(i0 + 2, n)))
    for i0 in range(n):
        for i1 in range(abs(i0 + n + 0)):
            qc.x(int(Mod(i1 + 1, n)))
    for i0 in range(n):
        for i1 in range(abs(n + i0 + 1)):
            qc.x(int(Mod(i1, n)))
    for i0 in range(n):
        for i1 in range(abs(n - n - 1)):
            qc.h(int(Mod(4, n)))
    for i0 in range(n):
        qc.ry(pi / (i0 + n + 1), int(Mod(i0, n)))
    for i0 in range(n):
        qc.rx(pi / (2 ** n + n + 1), int(Mod(i0 - 1, n)))
    return qc