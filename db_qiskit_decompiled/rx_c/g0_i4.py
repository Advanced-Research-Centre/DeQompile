from qiskit import QuantumCircuit
from math import pi
from sympy import Mod
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        for i1 in range(abs(n + i0 + 1)):
            qc.rz(pi / (2 ** (i0 - n) + 4 * n), int(Mod(5, n)))
    for i0 in range(n):
        for i1 in range(abs(n - n - 0)):
            qc.rz(pi / (2 ** n + i0 + n), int(Mod(i0, n)))
    for i0 in range(n):
        qc.rx(pi / (2 ** i0 + i0), int(0))
    return qc