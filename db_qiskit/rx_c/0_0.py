from qiskit import QuantumCircuit
from math import pi
import numpy as np
import random

def quantum_algorithm(n):
    qc = QuantumCircuit(n)
    for i0 in range(n):
        for i1 in range(abs(n - i0 + 1)):
            qc.rx(pi / (2 ** (2 * n) + n + 1), Mod(i1 - 1, n))
    return qc