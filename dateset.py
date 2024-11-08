from __future__ import annotations
import os
import numpy as np
import random
from fractions import Fraction
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister, qasm2
from qiskit.circuit.library import QFT
from qiskit_algorithms import Grover
from qiskit.circuit.library import GroverOperator
from math import pi

def h_0(n) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(0)
    return qc
    
def h_c(n) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    return qc

def gen_ghz(n) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1,n):
        qc.cx(0, i)
    return qc

def rx_c(n) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    angle = pi
    for i in range(n):
        qc.rx(angle, i)
        angle /= 2
    return qc
    
def rx_gradually_c(n) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    angle = pi
    for level in range(1, n + 1):
        current_angle = angle / (2 ** (level - 1))
        for i in range(level):
            qc.rx(current_angle, i)
    return qc

def qft(num_qubits: int) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Quantum Fourier Transform algorithm.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    """
    q = QuantumRegister(num_qubits, "q")
    c = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(q, c, name="qft")
    qc.compose(QFT(num_qubits=num_qubits), inplace=True)
    qc.measure_all()
    return qc

def qpe(num_qubits: int) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Quantum Phase Estimation algorithm for a phase which can be
    exactly estimated.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    """

    num_qubits = num_qubits - 1  # because of ancilla qubit
    q = QuantumRegister(num_qubits, "q")
    psi = QuantumRegister(1, "psi")
    c = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(q, psi, c, name="qpeexact")

    # get random n-bit string as target phase
    random.seed(10)
    theta = 0
    while theta == 0:
        theta = random.getrandbits(num_qubits)
    lam = Fraction(0, 1)
    # print("theta : ", theta, "correspond to", theta / (1 << n), "bin: ")
    for i in range(num_qubits):
        if theta & (1 << (num_qubits - i - 1)):
            lam += Fraction(1, (1 << i))

    qc.x(psi)
    qc.h(q)

    for i in range(num_qubits):
        angle = (lam * (1 << i)) % 2
        if angle > 1:
            angle -= 2
        if angle != 0:
            qc.cp(angle * np.pi, psi, q[i])

    qc.compose(
        QFT(num_qubits=num_qubits, inverse=True),
        inplace=True,
        qubits=list(range(num_qubits)),
    )
    qc.barrier()
    qc.measure(q, c)

    return qc

def grover(num_qubits: int, ancillary_mode: str = "noancilla") -> QuantumCircuit:
    """Returns a quantum circuit implementing Grover's algorithm.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    ancillary_mode -- defining the decomposition scheme
    """

    num_qubits = num_qubits - 1  # -1 because of the flag qubit
    q = QuantumRegister(num_qubits, "q")
    flag = AncillaRegister(1, "flag")

    state_preparation = QuantumCircuit(q, flag)
    state_preparation.h(q)
    state_preparation.x(flag)

    oracle = QuantumCircuit(q, flag)
    oracle.mcp(np.pi, q, flag)

    operator = GroverOperator(oracle, mcx_mode=ancillary_mode)
    iterations = Grover.optimal_num_iterations(1, num_qubits)

    num_qubits = operator.num_qubits - 1  # -1 because last qubit is "flag" qubit and already taken care of

    # num_qubits may differ now depending on the mcx_mode
    q2 = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q2, flag, name="grover")
    qc.compose(state_preparation, inplace=True)

    qc.compose(operator.power(iterations), inplace=True)
    qc.measure_all()
    qc.name = qc.name + "-" + ancillary_mode

    return qc

def db_qasm_generator(algorithm_name, max_qubit):
    
    directory = os.path.join('db_qasm_true')
    if not os.path.exists(directory):
        os.makedirs(directory)

    circuit_func = None
    # Patterns
    if algorithm_name == "h_0":
        circuit_func = h_0
    elif algorithm_name == "h_c":
        circuit_func = h_c
    elif algorithm_name == "gen_ghz":
        circuit_func = gen_ghz
    elif algorithm_name == "rx_c":
        circuit_func = rx_c
    elif algorithm_name == "rx_gradually_c":
        circuit_func = rx_gradually_c
    # Algorithms
    elif algorithm_name == "qft":
        circuit_func = qft
    elif algorithm_name == "qpe":
        circuit_func = qpe
    elif algorithm_name == "grover":
        circuit_func = grover
    # Others
    else:
        print("Unsupported algorithm.")
        return
    
    for n in range(2, max_qubit + 1):
        circuit = circuit_func(n)
        qasm_str = qasm2.dumps(circuit)
        filename = os.path.join(directory, f"{algorithm_name}_q{n}.qasm")
        with open(filename, "w") as file:
            file.write(qasm_str)
        print(f"Saved {filename}")

if __name__ == "__main__":
    # db_qasm_generator('h_0', 40)
    # db_qasm_generator('h_c', 40)
    db_qasm_generator('gen_ghz', 40)
    # db_qasm_generator('rx_c', 40)
    # db_qasm_generator('rx_gradually_c', 40)
    # db_qasm_generator('qft', 20)
    # db_qasm_generator('qpe', 20)
    # db_qasm_generator('grover', 15)