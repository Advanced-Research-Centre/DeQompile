from q_ast_gen import random_qiskit_ast_generator

import os
import time
from tqdm import tqdm
import ast
import shutil
from qiskit import QuantumCircuit, qasm2
import Levenshtein
import random
from copy import deepcopy
from qiskit.circuit import CircuitError
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import seaborn as sns
import numpy as np

class gp_deqompiler:
    """
    Genetic Programming based QASM-Qiskit Decompiler
    Uses Qiskit AST Generator to generate initial population
    """
    def __init__(self, algorithm_name, qubit_limit=20, generations=100, pop_size=50, max_length=10, max_loop_depth=2, operations=['h', 'x', 'cx'], compare_method='l_by_l', perform_crossover=True, crossover_rate=0.3, selection_method='tournament', perform_mutation=True, mutation_rate=0.1, new_gen_rate=0.2):

        self.algorithm_name = algorithm_name # Name of the algorithm being decompiled
        self.qubit_limit = qubit_limit  # Maximum number of qubits to consider (size of training set)
        self.generations = generations # Number of generations to run the GP
        self.pop_size = pop_size # Population size
        self.max_length = max_length # Maximum number of nodes of the AST for generated individuals
        self.crossover_rate=crossover_rate # Rate of crossover operation
        self.mutation_rate=mutation_rate # Rate of mutation operation
        self.new_gen_rate=new_gen_rate 
        self.max_loop_depth=max_loop_depth
        self.perform_crossover = perform_crossover # Perform crossover operation
        self.compare_method=compare_method
        self.perform_mutation = perform_mutation # Perform mutation operation
        self.selection_method = selection_method 
        self.operations=operations # List of quantum operations to consider in the AST
        self.db_qiskit_path = os.path.join('db_qiskit_decompiled', self.algorithm_name) # Path to save the generated Qiskit codes (unparsed ASTs) for each generation's, each individual
        os.makedirs(self.db_qiskit_path, exist_ok=True) # Create the directory if it does not exist
        self.db_qasm_path = os.path.join('db_qasm_decompiled_tmp') # Path to temporary save the generated QASM codes for each problem size for evaluating fitness for an individual
        os.makedirs(self.db_qasm_path, exist_ok=True) # Create the directory if it does not exist

        self.log_scores = []  # Log of scores for each generation

    def save_qiskit(self, population, generation):
        """
        Iterate over the population and save each individual's Python-Qiskit code to a file
        """
        for index, individual in enumerate(population):
            # Convert AST to Python code
            qiskit_code = ast.unparse(individual)
            # Create the filename, including the algorithm name and index
            filename = os.path.join(self.db_qiskit_path, f"g{generation}_i{index}.py")
            # Write Python code to the file
            with open(filename, 'w') as file:
                file.write(qiskit_code)

    def save_qasm(self, generation, index):

        decompiled_file_name = os.path.join(self.db_qiskit_path, f"g{generation}_i{index}.py")
        with open(decompiled_file_name, 'r') as file:
            module_code = file.read()
        local_namespace = {}
        exec(module_code, local_namespace)

        for i in range(2, self.qubit_limit + 1):
            try:
                qc = local_namespace['quantum_algorithm'](i)

                # # Handle if target and control same
                # modified_circuit = QuantumCircuit(i)
                # for ins in qc.data:
                #     gate, qargs, cargs = ins.operation, ins.qubits, ins.clbits
                #     if gate.name == 'cx':
                #         control_qubit, target_qubit = qargs
                #         if control_qubit._index != target_qubit._index:
                #             modified_circuit.cx(control_qubit, target_qubit)
                #         else:
                #             print("TBD: if CircuitError, this line is never reached")
                #             # Option 1: Remove conflicting control and make U-gate
                #             modified_circuit.x(target_qubit)                 
                #             # Option 2: Remove gate
                #             # Problem: Syntax error if only statement within for loop
                #             # Option 3: Adjust target qubit index to be different from control qubit index
                #             # target_qubit = qc.qubits[(target_qubit.index + 1) % qc.num_qubits]
                #             # modified_circuit.cx(control_qubit, target_qubit)
                #     else:
                #         modified_circuit.append(gate, qargs, cargs)
                # qasm_output = qasm2.dumps(modified_circuit)  
            except (ZeroDivisionError) as e:
                qasm_output = ""  # Save an empty QASM file if there's an error
                # print("ZeroDivisionError in g",generation," i",index," q",i," error:",e)
                # TBD: Division by zero occuring a lot for angle expr
            except (CircuitError) as e:
                qasm_output = ""  # Save an empty QASM file if there's an error
                # print("CircuitError in g",generation," i",index," q",i," error:",e)
                # TBD: Duplicate qubit argument occuring a lot for cx in GHZ
                # TBD: REGEX change CX in module_code and execute again
            else:
                qasm_output = qasm2.dumps(qc) 
            finally:
                qasm_filename = os.path.join(self.db_qasm_path, f"q{i}.qasm")
                with open(qasm_filename, 'w') as f:
                    f.write(qasm_output)

    def qasm_to_unitary(self, qasm_file_path):
        # Read QASM file and create a quantum circuit
        with open(qasm_file_path, 'r') as file:
            qasm_str = file.read()
        
        quantum_circuit = QuantumCircuit.from_qasm_str(qasm_str)
        
        # Use Aer simulator to get the unitary matrix
        backend = Aer.get_backend('unitary_simulator')
        transpiled_circuit = transpile(quantum_circuit, backend)
        
        # Get the unitary matrix
        job = backend.run(transpiled_circuit)
        unitary_matrix = job.result().get_unitary(transpiled_circuit)
        
        return unitary_matrix

    def qasm_to_gate_sequence(self, qasm_file_path):
        # Read QASM file and create a quantum circuit
        with open(qasm_file_path, 'r') as file:
            qasm_str = file.read()
        
        quantum_circuit = QuantumCircuit.from_qasm_str(qasm_str)
        
        # Extract gate sequence
        gate_sequence = []
        for instruction in quantum_circuit.data:
            gate_name = instruction.operation.name
            qubits = [qubit._index for qubit in instruction.qubits]
            if gate_name in ['rx', 'ry', 'rz']:
                params = [param for param in instruction.operation.params]
                gate_sequence.append((gate_name, tuple(qubits), params))
            else:
                gate_sequence.append((gate_name, tuple(qubits)))

        return gate_sequence

    def gate_sequence_similarity(self, seq1, seq2):
        seq1_str = ' '.join([f"{gate[0]}{gate[1]}{[f'{param:.6f}' for param in gate[2]]}" if len(gate) == 3 else f"{gate[0]}{gate[1]}" for gate in seq1])
        seq2_str = ' '.join([f"{gate[0]}{gate[1]}{[f'{param:.6f}' for param in gate[2]]}" if len(gate) == 3 else f"{gate[0]}{gate[1]}" for gate in seq2])
        ### Debugging line
        # print(f"Sequence 1: {seq1_str}")
        # print(f"Sequence 2: {seq2_str}")
        max_len = max(len(seq1_str), len(seq2_str))
        if max_len == 0:
            return 1.0
        return 1 - Levenshtein.distance(seq1_str, seq2_str) / max_len

    def gate_frequency_similarity(self, qasm_file_path1, qasm_file_path2):
        def get_gate_frequencies(qasm_file_path):
            with open(qasm_file_path, 'r') as file:
                qasm_str = file.read()
            
            quantum_circuit = QuantumCircuit.from_qasm_str(qasm_str)
            gate_count = {}
            for instruction in quantum_circuit.data:
                gate_name = instruction[0].name
                if gate_name in gate_count:
                    gate_count[gate_name] += 1
                else:
                    gate_count[gate_name] = 1
            return gate_count

        freq1 = get_gate_frequencies(qasm_file_path1)
        freq2 = get_gate_frequencies(qasm_file_path2)
        
        all_gates = set(freq1.keys()).union(set(freq2.keys()))
        vec1 = [freq1.get(gate, 0) for gate in all_gates]
        vec2 = [freq2.get(gate, 0) for gate in all_gates]
        
        dot_product = sum([vec1[i] * vec2[i] for i in range(len(all_gates))])
        norm1 = sum([x ** 2 for x in vec1]) ** 0.5
        norm2 = sum([x ** 2 for x in vec2]) ** 0.5
        
        return dot_product / (norm1 * norm2)

    def compare_qasm(self, qasm, target_qasm):
        def is_file_empty(file_path):
            return os.path.getsize(file_path) == 0

        if is_file_empty(qasm) or is_file_empty(target_qasm):
            return 0

        try:
            
            if self.compare_method == 'fidelity':
                # Calculate unitary matrices for both QASM files
                unitary1 = self.qasm_to_unitary(qasm)
                unitary2 = self.qasm_to_unitary(target_qasm)
                # Calculate fidelity
                score = process_fidelity(unitary1,unitary2)

            elif self.compare_method == 'seq_similarity':
                # Gate sequence similarity
                seq1 = self.qasm_to_gate_sequence(qasm)
                seq2 = self.qasm_to_gate_sequence(target_qasm)
                score = self.gate_sequence_similarity(seq1, seq2)

            elif self.compare_method == 'freq_similarity':
                # Gate frequency similarity
                score = self.gate_frequency_similarity(qasm, target_qasm)

            elif self.compare_method == 'combined':
                # Gate sequence similarity
                seq1 = self.qasm_to_gate_sequence(qasm)
                seq2 = self.qasm_to_gate_sequence(target_qasm)
                seq_similarity = self.gate_sequence_similarity(seq1, seq2)
                # Gate frequency similarity
                freq_similarity = self.gate_frequency_similarity(qasm, target_qasm)
                # Intersection
                with open(qasm, 'r') as file1, open(target_qasm, 'r') as file2:
                    qasm_lines = file1.readlines()
                    target_qasm_lines = file2.readlines()
                    # Convert lists to sets for intersection calculation
                    qasm_lines_set = set(qasm_lines)
                    target_qasm_lines_set = set(target_qasm_lines)
                    # Calculate intersection
                    intersection = qasm_lines_set.intersection(target_qasm_lines_set)
                    # Calculate max length of the two files
                    max_length = max(len(qasm_lines), len(target_qasm_lines))
                    # Calculate similarity score as intersection over max length of the two files
                    inter_section_score = len(intersection) / max_length if max_length else 0
                # Combine scores with equal weighting
                score = (seq_similarity + freq_similarity + inter_section_score) / 3

            elif self.compare_method == 'l_by_l':
                with open(qasm, 'r') as file1, open(target_qasm, 'r') as file2:
                    qasm_lines = file1.readlines()[3:]  # start from 4th line
                    target_qasm_lines = file2.readlines()[3:]  
                qasm_index = 0
                target_index = 0
                matched_lines = 0
                while qasm_index < len(qasm_lines) and target_index < len(target_qasm_lines):
                    if qasm_lines[qasm_index].strip() == target_qasm_lines[target_index].strip():
                        matched_lines += 1
                        qasm_index += 1
                    else:
                        matched_lines -= 0.5
                    target_index += 1
                # Calculate similarity score based on the number of matched lines over total lines in target_qasm
                score = matched_lines / len(target_qasm_lines) if target_qasm_lines else 0

        except FileNotFoundError:
            print(f"Error: One of the files not found ({qasm} or {target_qasm}).")
            return 0
        except Exception as e:
            print(f"Error comparing QASM files: {str(e)}")
            return 0   
        return score

    def est_description_complexity(self, generation, index):
        """
        Estimate description complexity of decompiled file (to score shorter programs higher)
        """
        # Count number of nodes in AST.
        decompiled_file_name = os.path.join(self.db_qiskit_path, f"g{generation}_i{index}.py")
        with open(decompiled_file_name, 'r') as file:
            module_code = file.read()
        ast_length = len(ast.parse(module_code).body[-1].body)
        # Normalize by self.max_length
        penalty = - ast_length / (2*self.max_length)  # 2x because length can increase due to crossover
        return penalty

    def evaluate(self, generation, index):
        """
        Given a Qiskit AST, generate QASM code for different qubit counts and compare with target QASM code
        """
        # Save QASM output for each individual for range of n values
        self.save_qasm(generation, index)
        # Calculate score for each QASM file
        scores = []
        for i in range(2, self.qubit_limit+1):
            decompiler_qasm_file = os.path.join(self.db_qasm_path, f"q{i}.qasm")
            true_qasm_file = os.path.join('db_qasm_true', f"{self.algorithm_name}_q{i}.qasm")
            scores.append(self.compare_qasm(decompiler_qasm_file, true_qasm_file))
        # Set base score to length of decompiled file (to force shorter programs)
        kc = self.est_description_complexity(generation, index)
        
        # Return the average score
        return kc + (sum(scores) / len(scores)) if scores else 0
    
    def clear_files(self, dir):
        """
        Clear all files (and subdirectories) in the target directory before saving new files
        """
        for content in os.listdir(dir):
            content_path = os.path.join(dir, content)
            try:
                # if os.path.isfile(content_path) or os.path.islink(content_path):  # Remove file
                os.unlink(content_path)         
                # elif os.path.isdir(content_path):                                 # Remove directory
                    # shutil.rmtree(content_path)     
            except Exception as e:
                print(f'Failed to delete {content_path}. Reason: {e}')

    def mutate(self, ast_circuit):
        # Create a deep copy of the AST to avoid modifying the original AST
        ast_circuit_copy = deepcopy(ast_circuit)

        # Randomly select mutation position
        num_operations = len(ast_circuit_copy.body[1].body) - 1 # Last element is return (thus -1)
        mutation_index = random.randint(1, num_operations - 1)
        
        # Generate level 1 AST (gate, for, nested-for), and strip of the qc defn (loc -3) and return (loc -1)
        new_sub_ast = random_qiskit_ast_generator(operations=self.operations, max_num_nodes=1, max_loop_depth=self.max_loop_depth).body[1].body[-2]  
            
        # Randomly select mutation type
        mutation_type = random.choice(['insert', 'modify'])
        
        if mutation_type == 'insert':
            # Insert the new sub-ast
            ast_circuit_copy.body[1].body.insert(mutation_index, new_sub_ast)
            
        elif mutation_type == 'modify':
            # Modify an existing quantum gate
            # existing_gate = ast_circuit_copy.body[1].body[mutation_index]
            # Replace the AST
            ast_circuit_copy.body[1].body[mutation_index] = new_sub_ast  
        
        ast.fix_missing_locations(ast_circuit_copy)
        return ast_circuit_copy

    def roulette_wheel_selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        selected_indices = random.choices(range(len(population)), weights=probabilities, k=2)
        return population[selected_indices[0]], population[selected_indices[1]]

    def tournament_selection(self, population, fitness_scores, k=3):
        selected_indices = random.sample(range(len(population)), k)
        selected_individuals = [(fitness_scores[i], population[i]) for i in selected_indices]
        parent1 = max(selected_individuals, key=lambda x: x[0])[1]
        parent2 = max(selected_individuals, key=lambda x: x[0])[1]
        return parent1, parent2

    def rank_selection(self, population, fitness_scores):
        sorted_population = sorted(zip(fitness_scores, population), key=lambda x: x[0])
        rank_probabilities = [(i + 1) / len(sorted_population) for i in range(len(sorted_population))]
        selected_indices = random.choices(range(len(population)), weights=rank_probabilities, k=2)
        return sorted_population[selected_indices[0]][1], sorted_population[selected_indices[1]][1]

    def random_selection(self, population):
        parent1, parent2 = random.sample(population, 2)
        return parent1, parent2

    def weighted_roulette_wheel_selection(self, population, fitness_scores, weight=2.0):
        total_fitness = sum(fitness_scores)
        weighted_fitness = [score ** weight for score in fitness_scores]
        total_weighted_fitness = sum(weighted_fitness)
        probabilities = [wf / total_weighted_fitness for wf in weighted_fitness]
        selected_indices = random.choices(range(len(population)), weights=probabilities, k=2)
        return population[selected_indices[0]], population[selected_indices[1]]

    def select_parents(self, population, fitness_scores, selection_method='tournament', k=3):
        if selection_method == 'roulette':
            return self.roulette_wheel_selection(population, fitness_scores)
        elif selection_method == 'tournament':
            return self.tournament_selection(population, fitness_scores, k)
        elif selection_method == 'rank':
            return self.rank_selection(population, fitness_scores)
        elif selection_method == 'random':
            return self.random_selection(population)
        elif selection_method == 'weighted_roulette':
            return self.weighted_roulette_wheel_selection(population, fitness_scores)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

    def crossover(self, parent1, parent2):
        
        # Note: body[0] --> import part, body[1] --> function, body[1].body --> level 1 inside function
        # print("\n",parent1.body[0],"\n",parent1.body[1],"\n",parent1.body[1].body)

        # Select crossover points
        index1 = random.randint(1, len(parent1.body[1].body) - 2)
        index2 = random.randint(1, len(parent2.body[1].body) - 2)
        
        # Swap subcircuits
        new_body1 = parent1.body[1].body[:index1] + parent2.body[1].body[index2:]
        new_body2 = parent2.body[1].body[:index2] + parent1.body[1].body[index1:]
        
        # Construct new ASTs
        child1 = ast.Module(body=[parent1.body[0], ast.FunctionDef(
            name=parent1.body[1].name, 
            args=parent1.body[1].args, 
            body=new_body1, 
            decorator_list=[]
        )], type_ignores=[])
        ast.fix_missing_locations(child1)
        
        child2 = ast.Module(body=[parent2.body[0], ast.FunctionDef(
            name=parent2.body[1].name, 
            args=parent2.body[1].args, 
            body=new_body2, 
            decorator_list=[]
        )], type_ignores=[])
        ast.fix_missing_locations(child2)
        
        return child1, child2

    def run(self):
        
        best_score = float('-inf')
        best_individual = None
        best_generation_index = -1
        best_individual_index = -1

        # Generate initial population once at the beginning
        population = []
        for _ in range(self.pop_size):
            ast_circuit = random_qiskit_ast_generator(operations=self.operations,max_num_nodes=self.max_length,max_loop_depth=self.max_loop_depth)
            population.append(ast_circuit)

        self.clear_files(self.db_qiskit_path)

        for generation in range(self.generations):
            start_time = time.time()
            print("\nGeneration:",generation,"\n")

            # Save the current population state
            self.save_qiskit(population, generation)

            # Evaluate fitness for each individual
            fitness_scores = [self.evaluate(generation, index) for index in range(0, self.pop_size)]
            # print("Fitness scores:", fitness_scores)  # Debugging line

            # Sort population by fitness (descending order)
            sorted_population = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
            sorted_scores, next_generation = zip(*sorted_population)
            next_generation = list(next_generation)
            sorted_scores = list(sorted_scores)
            self.log_scores.append(sorted_scores)

            # Select the best individual and corresponding score
            best_individual = next_generation[0]    # Ok to overwrite as elite members are preserved over generations
            best_score = sorted_scores[0]
            print("\tBest score in this generation:",best_score)
            print("\tAverage score in this generation:",sum(fitness_scores)/self.pop_size)
            
            # Find the index of the best individual in the original population
            best_individual_index = fitness_scores.index(best_score) # Will always be present in last generation due to elitism
            
            # # If the best score is 1, stop the iteration
            # if best_score == 1:
            #     break
            # Score is always less than 1 even for perfect match due to added kc penalty

            # Number of individuals to be generated by each method
            crossover_count = int(self.pop_size * self.crossover_rate) if self.perform_crossover == True else 0
            mutation_count = int(self.pop_size * self.mutation_rate) if self.perform_mutation == True else 0
            new_gen_count = int(self.pop_size * self.new_gen_rate)
            elite_count = self.pop_size - crossover_count - mutation_count - new_gen_count

            # Preserve elite individuals
            new_population = []
            new_population.extend(next_generation[:elite_count])

            # Apply crossover to generate new individuals
            while len(new_population) < elite_count + crossover_count:
                parent1, parent2 = self.select_parents(next_generation, sorted_scores, self.selection_method)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])

            # Apply mutation to new individuals
            for _ in range(mutation_count):
                individual_to_mutate = random.choice(new_population)
                new_population.append(self.mutate(individual_to_mutate))
            
            # Generate new individuals
            for _ in range(new_gen_count):
                ast_circuit = random_qiskit_ast_generator(operations=self.operations,max_num_nodes=self.max_length,max_loop_depth=self.max_loop_depth)
                new_population.append(ast_circuit)

            population = new_population

            end_time = time.time()
            time_taken = end_time - start_time
            tqdm.write(f"\tGeneration {generation + 1}/{self.generations} completed in {time_taken:.2f} seconds")

        # Clear temporary qasm files for last generation's last individual. 
        # These can be reconstructed if required from the decompiled qiskit files stored for all individuals of all generations.
        self.clear_files(self.db_qasm_path)

        # Unparse the AST of the best individual if found
        best_code = ast.unparse(best_individual) if best_individual else "No best individual found"

        return best_code, best_score, best_individual_index

    def save_results(self):
        pass

    def plot_results(self):

        plt.figure(figsize=(10, 6))
        
        # Define the color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # for algorithm, scores in all_scores.items():

        # Convert to numpy array for easier manipulation
        scores = np.array(self.log_scores)

        # Calculate mean, standard deviation, and max scores
        mean_scores = np.mean(scores, axis=1)
        # std_scores = np.std(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        # Apply smoothing filter
        smoothed_mean_scores = uniform_filter1d(mean_scores, size=3)
        # smoothed_std_scores = uniform_filter1d(std_scores, size=3)
        smoothed_max_scores = uniform_filter1d(max_scores, size=3)

        # Get the color for the current algorithm
        color = color_cycle.pop(0) if color_cycle else 'blue'

        # Plot the mean scores
        plt.plot(range(1, self.generations + 1), smoothed_mean_scores, label=f'Mean best fitness ({self.algorithm_name})', color=color)
            
        # Plot the max scores with the same color
        plt.plot(range(1, self.generations + 1), smoothed_max_scores, label=f'Max fitness ({self.algorithm_name})', linestyle='--', color=color)

        # Fill between mean scores and max scores
        plt.fill_between(range(1, self.generations + 1), smoothed_mean_scores, smoothed_max_scores, alpha=0.2, color=color)

        # Add titles and labels
        plt.title('DeQompile performance over generations')
        plt.xlabel('Generation')
        plt.ylabel(f'Fitness (method: {self.compare_method})')
        plt.legend()
        plt.grid(True)

        # Save the plot if a save path is provided
        # if save_path:
        #     plt.savefig(save_path)
        
        # Show the plot
        plt.show()

    @staticmethod
    def get_operations(algorithm_name, qubit_limit):
        """
        Extract operations from target qasms
        """
        operations = []
        for i in range(2, qubit_limit+1):
            qasm_file_path = os.path.join('db_qasm_true', f"{algorithm_name}_q{i}.qasm")
            with open(qasm_file_path, 'r') as file:
                qasm_str = file.read()
            quantum_circuit = QuantumCircuit.from_qasm_str(qasm_str)
            for instruction in quantum_circuit.data:
                gate_name = instruction.operation.name
                if gate_name not in operations:
                    operations.append(gate_name)
        return operations

if __name__ == "__main__":

    algorithm_name = 'rx_c'                  # Required. Supports: {'h_0', 'h_c', 'gen_ghz', 'rx_c', 'rx_gradually_c', 'qft', 'qpe', 'grover'}
    qubit_limit = 20                            # Default: 20
    
    generations = 150                           # Default: 100
    pop_size = 40                              # Default: 50
    
    max_length = 10                             # Default: 10
    max_loop_depth = 2                          # Default: 2
    # operations = ['h', 'x', 'rx', 'ry', 'rz']   # Default: ['h', 'x', 'cx']. Supports: 'Gate' where <QuantumCircuit>.<Gate>(args) is valid
    operations = gp_deqompiler.get_operations(algorithm_name, qubit_limit)
   
    compare_method = 'l_by_l'                   # Default: '1_by_1'. Supports: 'fidelity', 'seq_similarity', 'freq_similarity', 'combined', 'l_by_l'
    
    perform_crossover = True                    # Default: True
    crossover_rate = 0.3                        # Default: 0.3
    selection_method = 'tournament'             # Default: 'tournament'
    perform_mutation = True                     # Default: True
    mutation_rate = 0.2                         # Default: 0.2
    new_gen_rate = 0.2                          # Default: 0.2

    decompiler = gp_deqompiler(algorithm_name=algorithm_name,
                               qubit_limit=qubit_limit,
                               generations=generations,
                               pop_size=pop_size,
                               max_length=max_length,
                               max_loop_depth=max_loop_depth,
                               operations=operations,
                               compare_method=compare_method,
                               perform_crossover=perform_crossover,
                               crossover_rate=crossover_rate,
                               selection_method=selection_method,
                               perform_mutation=perform_mutation,
                               mutation_rate=mutation_rate,
                               new_gen_rate=new_gen_rate)
    
    best_code, best_score, best_individual_index = decompiler.run()
    
    print("\n\nBest code:\n")
    print(best_code,"\n")
    print("\nBest score:",best_score)
    print("Best individial ID in last generation:",best_individual_index)

    decompiler.plot_results()