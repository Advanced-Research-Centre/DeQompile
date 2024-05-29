import ast
from tqdm import tqdm
import subprocess
import os
import shutil
import importlib.util
from circuit_generation import *
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Operator, state_fidelity
import Levenshtein

class genetic_Decompiler:
    def __init__(self, algorithm_name, qubit_limit=20, generations=100, pop_size=50, max_length=10, perform_crossover=True,
                crossover_rate=0.3, new_gen_rate=0.2,mutation_rate=0.1,compare_method='l_by_l',
                  perform_mutation=True, selection_method='tournament',operations = ['h', 'x', 'cx']):
        self.algorithm_name = algorithm_name
        self.qubit_limit = qubit_limit
        self.generations = generations
        self.pop_size = pop_size
        self.max_length = max_length
        self.crossover_rate=crossover_rate
        self.mutation_rate=mutation_rate
        self.new_gen_rate=new_gen_rate
        self.perform_crossover = perform_crossover
        self.compare_method=compare_method
        self.perform_mutation = perform_mutation
        self.selection_method = selection_method
        self.operations=operations
        # Initialize the path for saving files related to the algorithm
        self.path = os.path.join('genetic_deQ', self.algorithm_name)
        self.qasm_path=os.path.join('genetic_deQ_qasm', self.algorithm_name)
        os.makedirs(self.path, exist_ok=True)  # Create the directory if it does not exist
        os.makedirs(self.qasm_path, exist_ok=True) 
 

    def generate_initial_population(self,size):
        population = []
        for _ in range(size):
            # num_qubits = random.randint(2, self.qubit_limit)
            num_nodes = random.randint(1, self.max_length)
            ast_circuit = generate_random_circuit_ast( num_nodes,self.operations)
            population.append(ast_circuit)
        return population

    def analyze_difference(self, sequence, target_sequence):
        differences = []
        for i, (gate1, gate2) in enumerate(zip(sequence, target_sequence)):
            if gate1 != gate2:
                differences.append((i, gate1, gate2))
        return differences

    def mutate_individual(self, individual, target_individual):
        sequence = self.qasm_to_gate_sequence(individual)
        target_sequence = self.qasm_to_gate_sequence(target_individual)
        differences = self.analyze_difference(sequence, target_sequence)

        mutation_type = random.choice(['operation', 'phase', 'coefficient'])
        for index, gate1, gate2 in differences:
            if mutation_type == 'operation':
                individual[index] = self.mutate_operation(gate1, gate2)
            elif mutation_type == 'phase':
                individual[index] = self.mutate_phase(gate1, gate2)
            elif mutation_type == 'coefficient':
                individual[index] = self.mutate_coefficient(gate1, gate2)
        return individual

    def mutate_operation(self, gate1, gate2):
        new_gate = random.choice(self.operations)
        return (new_gate, gate1[1], gate1[2]) if len(gate1) == 3 else (new_gate, gate1[1])

    def mutate_phase(self, gate1, gate2):
        new_phase = random.uniform(-3.14, 3.14)
        return (gate1[0], gate1[1], [new_phase])

    def mutate_coefficient(self, gate1, gate2):
        a = random.choice([random.uniform(-3.14, 3.14), 0])
        b = random.choice([random.uniform(-3.14, 3.14), 0])
        c = random.randint(0, 4)
        new_phase = 3.14 * 1 / (2**a + b + c)
        return (gate1[0], gate1[1], [new_phase])
        # Randomly select mutation position
        num_operations = len(ast_circuit.body[0].body) - 1
        mutation_index = random.randint(1, num_operations - 1)
        
        # Randomly select mutation type
        mutation_type = random.choice(['insert', 'modify'])
        
        if mutation_type == 'insert':
            # Insert a new quantum gate
            new_gate_ast = generate_random_circuit_ast(1, self.operations)  # Generate AST for one random gate
            ast_circuit.body[0].body.insert(mutation_index, new_gate_ast.body[0].body[1])  # Insert the new gate
            
        elif mutation_type == 'modify':
            # Modify an existing quantum gate
            existing_gate = ast_circuit.body[0].body[mutation_index]
            if isinstance(existing_gate, ast.Expr):
                new_gate_ast = generate_random_circuit_ast(1, self.operations)
                ast_circuit.body[0].body[mutation_index] = new_gate_ast.body[0].body[1]  # Replace the gate
        
        ast.fix_missing_locations(ast_circuit)
        return ast_circuit

    def crossover(self, parent1, parent2):
        # Select crossover points
        index1 = random.randint(1, len(parent1.body[0].body) - 2)
        index2 = random.randint(1, len(parent2.body[0].body) - 2)
        
        # Swap subcircuits
        new_body1 = parent1.body[0].body[:index1] + parent2.body[0].body[index2:]
        new_body2 = parent2.body[0].body[:index2] + parent1.body[0].body[index1:]
        
        # Construct new ASTs
        child1 = ast.Module(body=[ast.FunctionDef(
            name=parent1.body[0].name, 
            args=parent1.body[0].args, 
            body=new_body1, 
            decorator_list=[]
        )], type_ignores=[])
        
        child2 = ast.Module(body=[ast.FunctionDef(
            name=parent2.body[0].name, 
            args=parent2.body[0].args, 
            body=new_body2, 
            decorator_list=[]
        )], type_ignores=[])
        
        ast.fix_missing_locations(child1)
        ast.fix_missing_locations(child2)
        
        return child1, child2

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

    def save(self, population):
    # Clear all files in the target directory before saving new files
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        # Iterate over the population and save each individual's Python code to a file
        for index, individual in enumerate(population):
            # Convert AST to Python code
            python_code = ast.unparse(individual)

            # Create the filename, including the algorithm name and index
            filename = os.path.join(self.path, f"{self.algorithm_name}_{index}.py")
            
            # Write Python code to the file
            with open(filename, 'w') as file:
                file.write(python_code)

    def save_qasm(self):
     
        for filename in os.listdir(self.path):
            if filename.endswith('.py'):
                full_py_path = os.path.join(self.path, filename)
                
                # Read and modify the script as discussed above
                with open(full_py_path, 'r') as file:
                    module_code = "from qiskit import QuantumCircuit\nimport numpy as np\nimport random\nfrom math import pi\n" + file.read()

                
                local_namespace = {}
                exec(module_code, local_namespace)
                
                # Set up the directory for QASM files
                file_base_name = filename[:-3]  # Remove '.py' extension
                qasm_dir_path = os.path.join(self.qasm_path, file_base_name)
                os.makedirs(qasm_dir_path, exist_ok=True)
                
                # Generate QASM files for each qubit count
                for i in range(2, 41):
                    qc = local_namespace['generate_random_circuit_ast'](i)
                    qasm_output = qc.qasm()
                    qasm_filename = os.path.join(qasm_dir_path, f"{file_base_name}_{i}.qasm")
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
            gate_name = instruction[0].name
            qubits = [qubit.index for qubit in instruction[1]]
            if gate_name in ['rx', 'ry', 'rz']:
                params = [param for param in instruction[0].params]
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
        try:
            if self.compare_method == 'intersection':
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
                    score = len(intersection) / max_length if max_length else 0
                    return score

            elif self.compare_method == 'fidelity':
                # Calculate unitary matrices for both QASM files
                unitary1 = self.qasm_to_unitary(qasm)
                unitary2 = self.qasm_to_unitary(target_qasm)
                
                # Calculate fidelity
                fidelity_score = state_fidelity(Operator(unitary1), Operator(unitary2))
                return fidelity_score

            elif self.compare_method == 'seq_similarity':
                # Gate sequence similarity
                seq1 = self.qasm_to_gate_sequence(qasm)
                seq2 = self.qasm_to_gate_sequence(target_qasm)
                seq_similarity = self.gate_sequence_similarity(seq1, seq2)
                return seq_similarity

            elif self.compare_method == 'freq_similarity':
                # Gate frequency similarity
                freq_similarity = self.gate_frequency_similarity(qasm, target_qasm)
                return freq_similarity

            elif self.compare_method == 'combined':
                # Gate sequence similarity
                seq1 = self.qasm_to_gate_sequence(qasm)
                seq2 = self.qasm_to_gate_sequence(target_qasm)
                seq_similarity = self.gate_sequence_similarity(seq1, seq2)
                
                # Gate frequency similarity
                freq_similarity = self.gate_frequency_similarity(qasm, target_qasm)
                
                # Unitary fidelity
                unitary1 = self.qasm_to_unitary(qasm)
                unitary2 = self.qasm_to_unitary(target_qasm)
                fidelity_score = state_fidelity(Operator(unitary1), Operator(unitary2))
                
                # Combine scores with equal weighting
                combined_score = (seq_similarity + freq_similarity + fidelity_score) / 3
                return combined_score

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
                return score

        except FileNotFoundError:
            print(f"Error: One of the files not found ({qasm} or {target_qasm}).")
            return 0
        except Exception as e:
            print(f"Error comparing QASM files: {str(e)}")
            return 0   

    def evaluate(self, individual, individual_index):
        qasm_dir = os.path.join(self.qasm_path, f"{self.algorithm_name}_{individual_index}")
        target_qasm_dir = "Circuits"
        
        # Calculate score for each QASM file
        scores = []
        for i in range(2, 41):
            qasm_file = os.path.join(qasm_dir, f"{self.algorithm_name}_{individual_index}_{i}.qasm")
            target_qasm_file = os.path.join(target_qasm_dir, f"{self.algorithm_name}_{i}.qasm")
             ## debug
            # print(qasm_file,target_qasm_file)
            score = self.compare_qasm(qasm_file, target_qasm_file)
            scores.append(score)
        
        # Return the average score
        average_score = sum(scores) / len(scores) if scores else 0
        return average_score
            
    def run(self):
        if not self.perform_crossover and not self.perform_mutation:
            print("Warning: Both crossover and mutation are disabled; the population will not evolve.")

        best_score = float('-inf')
        best_individual = None
        best_generation_index = -1

        # Generate initial population once at the beginning
        population = self.generate_initial_population(self.pop_size)

        for generation in range(self.generations):
            start_time = time.time()

            # Save the current population state and QASM data
            self.save(population)
            self.save_qasm()

            # Evaluate fitness for each individual
            # [print(index) for index, individual in enumerate(population)]
            fitness_scores = [self.evaluate(individual, index) for index, individual in enumerate(population)]
            print("Fitness scores:", fitness_scores)  # Debugging line

             # Sort population by fitness (descending order)
            sorted_population = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
            sorted_scores, next_generation = zip(*sorted_population)
            next_generation = list(next_generation)
            sorted_scores = list(sorted_scores)

            # Select the best individual and corresponding score
            best_individual = next_generation[0]
            best_score = sorted_scores[0]
            
            # Find the index of the best individual in the original population
            best_individual_index = fitness_scores.index(best_score)
            
            # Print debugging information
            print(f"Generation {generation + 1}: Best score = {best_score}")

            # If the best score is 1, stop the iteration
            if best_score == 1:
                break
            new_population = []

            # Number of individuals to be generated by each method
            crossover_count = int(self.pop_size * self.crossover_rate)
            mutation_count = int(self.pop_size * self.mutation_rate)
            new_gen_count = int(self.pop_size * self.new_gen_rate)
            elite_count = self.pop_size - crossover_count - mutation_count - new_gen_count

            # Preserve elite individuals
            new_population.extend(next_generation[:elite_count])

            # Apply crossover to generate new individuals
            while len(new_population) < elite_count + crossover_count:
                parent1, parent2 = self.select_parents(next_generation, sorted_scores, self.selection_method)
                
                child1, child2 = self.crossover(parent1, parent2)
               
                new_population.extend([child1, child2])

            # Ensure the population size is correct after crossover
            new_population = new_population[:elite_count + crossover_count]

            # Apply mutation to new individuals
            for _ in range(mutation_count):
                if new_population:
                    individual_to_mutate = random.choice(new_population)
                    new_population.append(self.mutate_individual(individual_to_mutate))
            
            # Generate new individuals
            new_population.extend(self.generate_initial_population(new_gen_count))


            # Ensure the population size is correct after all operations
            new_population = new_population[:self.pop_size]

            population = new_population

            end_time = time.time()
            time_taken = end_time - start_time
            tqdm.write(f"Generation {generation + 1}/{self.generations} completed in {time_taken:.2f} seconds")

        # Unparse the AST of the best individual if found
        best_code = ast.unparse(best_individual) if best_individual else "No best individual found"
        return best_code, best_score, best_individual_index