from q_ast_gen import random_qiskit_ast_generator

import os
import time
from tqdm import tqdm
import ast
import shutil

class gp_deqompiler:
    """
    Genetic Programming based QASM-Qiskit Decompiler
    Uses Qiskit AST Generator to generate initial population
    """
    def __init__(self, algorithm_name, qubit_limit=20, generations=100, pop_size=50, max_length=10, perform_crossover=True,
                crossover_rate=0.3, new_gen_rate=0.2,mutation_rate=0.1,compare_method='l_by_l',max_loop_depth=2,
                  perform_mutation=True, selection_method='tournament',operations = ['h', 'x', 'cx']):
        self.algorithm_name = algorithm_name
        self.qubit_limit = qubit_limit
        self.generations = generations
        self.pop_size = pop_size
        self.max_length = max_length
        self.crossover_rate=crossover_rate
        self.mutation_rate=mutation_rate
        self.new_gen_rate=new_gen_rate
        self.max_loop_depth=max_loop_depth
        self.perform_crossover = perform_crossover
        self.compare_method=compare_method
        self.perform_mutation = perform_mutation
        self.selection_method = selection_method
        self.operations=operations
        # Initialize the path for saving files related to the algorithm
        self.path = os.path.join('db_qiskit', self.algorithm_name)
        self.qasm_path = os.path.join('db_qasm', self.algorithm_name)
        os.makedirs(self.path, exist_ok=True)  # Create the directory if it does not exist
        os.makedirs(self.qasm_path, exist_ok=True) 

    def save_qasm(self):
        for filename in os.listdir(self.path):
            if filename.endswith('.py'):
                full_py_path = os.path.join(self.path, filename)
                
                with open(full_py_path, 'r') as file:
                    module_code = file.read()

                local_namespace = {}
                exec(module_code, local_namespace)
                
                # Set up the directory for QASM files
                file_base_name = filename[:-3]  # Remove '.py' extension
                qasm_dir_path = os.path.join(self.qasm_path, file_base_name)
                os.makedirs(qasm_dir_path, exist_ok=True)
                
                # Generate QASM files for each qubit count
                for i in range(2, self.qubit_limit + 1):
                    try:
                        qc = local_namespace['generate_random_circuit_ast'](i)
                        
                        # Check and modify cx gates if necessary
                        modified_circuit = QuantumCircuit(qc.num_qubits)
                        for gate, qargs, cargs in qc.data:
                            if gate.name == 'cx':
                                control_qubit, target_qubit = qargs
                                if control_qubit.index == target_qubit.index:
                                    # Adjust target qubit index to be different from control qubit index
                                    target_qubit = qc.qubits[(target_qubit.index + 1) % qc.num_qubits]
                                    modified_circuit.cx(control_qubit, target_qubit)
                                else:
                                    modified_circuit.cx(control_qubit, target_qubit)
                            else:
                                modified_circuit.append(gate, qargs, cargs)
                        
                        qasm_output = modified_circuit.qasm()
                    except (CircuitError, ZeroDivisionError) as e:
                    # Handle both CircuitError and ZeroDivisionError
                    # print(f"Error generating QASM for {filename} with {i} qubits: {e}")
                        qasm_output = ""  # Save an empty QASM file if there's an error
                    
                    qasm_filename = os.path.join(qasm_dir_path, f"{file_base_name}_{i}.qasm")
                    with open(qasm_filename, 'w') as f:
                        f.write(qasm_output)

    def save_qiskit(self, population, generation):
        """
        Iterate over the population and save each individual's Python-Qiskit code to a file
        """
        for index, individual in enumerate(population):
            # Convert AST to Python code
            qiskit_code = ast.unparse(individual)
            # Create the filename, including the algorithm name and index
            filename = os.path.join(self.path, f"{generation}_{index}.py")
            # Write Python code to the file
            with open(filename, 'w') as file:
                file.write(qiskit_code)

    def clear_files(self):
        """
        Clear all files (and subdirectories) in the target directory before saving new files
        """
        for content in os.listdir(self.path):
            content_path = os.path.join(self.path, content)
            try:
                # if os.path.isfile(content_path) or os.path.islink(content_path):  # Remove file
                os.unlink(content_path)         
                # elif os.path.isdir(content_path):                                 # Remove directory
                    # shutil.rmtree(content_path)     
            except Exception as e:
                print(f'Failed to delete {content_path}. Reason: {e}')

    def run(self):
        
        best_score = float('-inf')
        best_individual = None
        best_generation_index = -1
        best_individual_index = -1
        best_code = ""

        # Generate initial population once at the beginning
        population = []
        for _ in range(self.pop_size):
            ast_circuit = random_qiskit_ast_generator(operations=self.operations,max_num_nodes=self.max_length,max_loop_depth=self.max_loop_depth)
            population.append(ast_circuit)

        self.clear_files()

        for generation in range(self.generations):
            start_time = time.time()

            # Save the current population state
            self.save_qiskit(population, generation)
            # Save QASM output for each individual for range of n values
            self.save_qasm()

        #     # Evaluate fitness for each individual
        #     # [print(index) for index, individual in enumerate(population)]
        #     fitness_scores = [self.evaluate(individual, index) for index, individual in enumerate(population)]
        #     print("Fitness scores:", fitness_scores)  # Debugging line

        #      # Sort population by fitness (descending order)
        #     sorted_population = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
        #     sorted_scores, next_generation = zip(*sorted_population)
        #     next_generation = list(next_generation)
        #     sorted_scores = list(sorted_scores)

        #     # Select the best individual and corresponding score
        #     best_individual = next_generation[0]
        #     best_score = sorted_scores[0]
            
        #     # Find the index of the best individual in the original population
        #     best_individual_index = fitness_scores.index(best_score)
            
        #     # Print debugging information
        #     print(f"Generation {generation + 1}: Best score = {best_score}")

        #     # If the best score is 1, stop the iteration
        #     if best_score == 1:
        #         break
        #     new_population = []

        #     # Number of individuals to be generated by each method
        #     crossover_count = int(self.pop_size * self.crossover_rate) if self.perform_crossover == True else 0
        #     mutation_count = int(self.pop_size * self.mutation_rate) if self.perform_mutation == True else 0
        #     new_gen_count = int(self.pop_size * self.new_gen_rate)
        #     elite_count = self.pop_size - crossover_count - mutation_count - new_gen_count

        #     # Preserve elite individuals
        #     new_population.extend(next_generation[:elite_count])

        #     # Apply crossover to generate new individuals
        #     while len(new_population) < elite_count + crossover_count:
        #         parent1, parent2 = self.select_parents(next_generation, sorted_scores, self.selection_method)
                
        #         child1, child2 = self.crossover(parent1, parent2)
               
        #         new_population.extend([child1, child2])

        #     # Apply mutation to new individuals
        #     for _ in range(mutation_count):
        #         if new_population:
        #             individual_to_mutate = random.choice(new_population)
        #             new_population.append(self.mutate(individual_to_mutate))
            
        #     # Generate new individuals
        #     new_population.extend(self.generate_initial_population(new_gen_count))


        #     # Ensure the population size is correct after all operations
        #     # new_population = new_population[:self.pop_size]

        #     population = new_population

            end_time = time.time()
            time_taken = end_time - start_time
            tqdm.write(f"Generation {generation + 1}/{self.generations} completed in {time_taken:.2f} seconds")

        # # Unparse the AST of the best individual if found
        # best_code = ast.unparse(best_individual) if best_individual else "No best individual found"
        return best_code, best_score, best_individual_index

if __name__ == "__main__":

    operations = ['h', 'x', 'rx', 'ry', 'rz']
    generations = 3
    algorithm_name = 'rx_c'
    compare_method = 'seq_similarity'
    perform_crossover = False
    perform_mutation = True
    pop_size = 8
    new_gen_rate = 0.6
    
    decompiler = gp_deqompiler(operations=operations,generations=generations,algorithm_name=algorithm_name,compare_method=compare_method,
                                perform_crossover=perform_crossover,perform_mutation=perform_mutation,pop_size=pop_size,new_gen_rate=new_gen_rate)
    
    best_code, best_score, best_generation_index = decompiler.run()
    
    print(best_code, '\n', best_score, '\n',best_generation_index)
    # 