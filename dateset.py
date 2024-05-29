import os
print(os.getcwd())
from algorthims import *


def QASM_generator(circuitname, max_qubit):
    # Create dir
    directory = "Circuits"
    if not os.path.exists(directory):
        os.makedirs(directory)

    circuit_func = None
    if circuitname == "grover":
        circuit_func = grover
    elif circuitname == "qft":
        circuit_func = qft
    elif circuitname == "qpe":
        circuit_func = qpe
    elif circuitname == "h_c":
        circuit_func = h_c
    elif circuitname == "rx_c":
        circuit_func = rx_c
    elif circuitname == "rx_gradually_c":
        circuit_func = rx_gradually_c
    elif circuitname == "h_0":
        circuit_func = h_0
    else:
        print("Unsupported circuit name.")
        return
    for n in range(2, max_qubit + 1):
        circuit = circuit_func(n)

        qasm_str = circuit.qasm()

        filename = os.path.join(directory, f"{circuitname}_{n}.qasm")
        with open(filename, "w") as file:
            file.write(qasm_str)
        print(f"Saved {filename}")


    if __name__ == "main":
        # QASM_generator('qft',20)
        # QASM_generator('qpe',20)
        # QASM_generator('grover',20)
        # QASM_generator('h_c',40)
        # QASM_generator('rx_c',40)
        # QASM_generator('rx_gradually_c',40)
        QASM_generator('h_0', 40)

