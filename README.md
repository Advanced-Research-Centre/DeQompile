# Quantum Circuit Genetic Decompiler

The Quantum Circuit Genetic Decompiler project provides a modular approach to evolving quantum circuits using genetic programming techniques. This project is split into four main parts, each responsible for handling different aspects of the quantum circuit decompilation and optimization process.

## Components

- **dataset.py**: Manages the loading, preprocessing, and handling of quantum circuit datasets. This module ensures that data is formatted and ready for use in the decompilation process.
- **circuit_generation.py**: Responsible for generating random quantum circuits. This module serves as the core for creating initial circuit conditions and potential solutions that the genetic algorithm can evolve.
- **decompiler.py**: Contains the logic for the genetic algorithm, including mutation, crossover, and selection mechanisms tailored for optimizing quantum circuits.
- **main.py**: The entry point of the application, orchestrating the flow between dataset management, circuit generation, and the decompilation process. It initializes the process, executes the genetic algorithm, and outputs the results.



