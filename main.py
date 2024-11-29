from decomplier import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
import json
import os


class GeneticDecompilationPlotter:
    def __init__(self, algorithms, algorithm_name, qubit_limit=20, generations=100, pop_size=50, max_length=10, 
                 perform_crossover=True, crossover_rate=0.3, new_gen_rate=0.2, mutation_rate=0.1, 
                 compare_method='l_by_l', max_loop_depth=2, mutation_rate_2=0.5, perform_annealing=True,
                 perform_mutation=True, selection_method='tournament', operations=['h', 'x', 'cx'], 
                 rep=10):
        
        self.algorithms = algorithms
        self.algorithm_name = algorithm_name
        self.qubit_limit = qubit_limit
        self.generations = generations
        self.pop_size = pop_size
        self.max_length = max_length
        self.perform_crossover = perform_crossover
        self.crossover_rate = crossover_rate
        self.new_gen_rate = new_gen_rate
        self.mutation_rate = mutation_rate
        self.compare_method = compare_method
        self.max_loop_depth = max_loop_depth
        self.mutation_rate_2 = mutation_rate_2
        self.perform_annealing = perform_annealing
        self.perform_mutation = perform_mutation
        self.selection_method = selection_method
        self.operations = operations
        self.rep = rep

    def run_experiments(self):
        best_scores_list = {alg: [] for alg in self.algorithms}
        all_scores_list = {alg: [] for alg in self.algorithms}
        best_individual_list = {alg: [] for alg in self.algorithms}
        
        for algorithm in self.algorithms:
            best_list = []
            score_list = [] 
            individual_list = []
            
            for _ in range(self.rep):
                decompiler = genetic_Decompiler(
                    operations=self.operations,
                    generations=self.generations,
                    algorithm_name=algorithm,
                    compare_method=self.compare_method,
                    perform_crossover=self.perform_crossover,
                    perform_mutation=self.perform_mutation,
                    pop_size=self.pop_size,
                    new_gen_rate=self.new_gen_rate,
                    crossover_rate=self.crossover_rate,
                    mutation_rate=self.mutation_rate,
                    max_length=self.max_length,
                    qubit_limit=self.qubit_limit,
                    max_loop_depth=self.max_loop_depth,
                    mutation_rate_2=self.mutation_rate_2,
                    perform_annealing=self.perform_annealing,
                    selection_method=self.selection_method
                )
                operations = decompiler.get_quantum_gates_from_qasm()
                decompiler.operations = operations

                print(f'repetition {_} start:')
                print(f'gates included for this circuit: {decompiler.operations}')

                best_code, best_scores, all_scores = decompiler.run()
                best_list.append(best_scores)
                score_list.append(all_scores)
                individual_list.append(best_code)
            
            best_scores_list[algorithm] = best_list
            all_scores_list[algorithm] = score_list
            best_individual_list[algorithm] = individual_list
        
        return best_scores_list, all_scores_list, best_individual_list

    def save_results(self, filename, best_scores_list, all_scores_list, best_individual_list):
        # Ensure the data directory exists
        os.makedirs('data', exist_ok=True)

        # Combine all data into a single dictionary
        data = {
            'best_scores_list': best_scores_list,
            'all_scores_list': all_scores_list,
            'best_individual_list': best_individual_list
        }

        # Save the combined data as a JSON file
        with open(os.path.join('data', filename), 'w') as f:
            json.dump(data, f, indent=4)

    def load_data(self, filename):
        # Ensure the data directory exists
        filepath = os.path.join('data', filename)

        # Read the combined data from the JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract the individual components
        all_best_scores = data['best_scores_list']
        all_scores = data['all_scores_list']
        all_individual = data['best_individual_list']

        return all_best_scores, all_scores, all_individual

    def plot_scores(self, all_scores, save_path=None):
        plt.figure(figsize=(10, 6))
        
        # Define the color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for algorithm, scores in all_scores.items():
            # Convert to numpy array for easier manipulation
            scores = np.array(scores)

            # Calculate mean, standard deviation, and max scores
            mean_scores = np.mean(scores, axis=0)
            std_scores = np.std(scores, axis=0)
            max_scores = np.max(scores, axis=0)

            # Apply smoothing filter
            smoothed_mean_scores = uniform_filter1d(mean_scores, size=3)
            smoothed_std_scores = uniform_filter1d(std_scores, size=3)
            smoothed_max_scores = uniform_filter1d(max_scores, size=3)

            # Get the color for the current algorithm
            color = color_cycle.pop(0) if color_cycle else 'blue'

            # Plot the mean scores
            plt.plot(range(1, self.generations + 1), smoothed_mean_scores, label=f'Mean Best Score ({algorithm})', color=color)
            
            # Plot the max scores with the same color
            plt.plot(range(1, self.generations + 1), smoothed_max_scores, label=f'Max Score ({algorithm})', linestyle='--', color=color)

            # Fill between mean scores and max scores
            plt.fill_between(range(1, self.generations + 1), 
                             smoothed_mean_scores, 
                             smoothed_max_scores, 
                             alpha=0.2, color=color)

        # Add titles and labels
        plt.title('Genetic Decompilation Performance Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Score')
        plt.legend()
        plt.grid(True)

        # Save the plot if a save path is provided
        if save_path:
            plt.savefig(save_path)
        
        # Show the plot
        plt.show()




if __name__ == "__main__":

    #####  run decomplier    
        # decompiler = genetic_Decompiler(operations=['h', 'x', 'rx', 'ry', 'rz'],
        #                             generations=10,algorithm_name='rx_c',compare_method='seq_similarity',
        #                             perform_crossover=False,perform_mutation=True,pop_size=100,new_gen_rate=0.6)
        # best_code, best_score, best_generation_index = decompiler.run()
        # print(best_code, '\n', best_score, '\n',best_generation_index)
    #####




    #####   get plots


    algorithm = ['h_0', 'h_c', 'rx_c']
    plotter = GeneticDecompilationPlotter(algorithm_name='example', algorithms=algorithm, mutation_rate=0.3,
                                        new_gen_rate=0.3,crossover_rate=0.2,
                                            mutation_rate_2=0.99,max_length=3,max_loop_depth=2,qubit_limit=10,
                                        compare_method='combined',pop_size=40, generations=400, rep=3)

    # Run experiments
    all_best_scores, all_scores, all_individual = plotter.run_experiments()
    plotter.save_results('simple_test_2', all_best_scores, all_scores, all_individual)
    all_best_scores, all_scores, all_individual = plotter.load_data('simple_test_2')
    # Plot scores and save the figure
    plotter.plot_scores(all_best_scores, save_path='Figure/simple_test_2.png')


    # algorithm = ['ry_decomposed_rx_rz','ry_c', 'ry_decomposed']
    # plotter = GeneticDecompilationPlotter(algorithm_name='example', algorithms=algorithm, mutation_rate=0.3,
    #                                     new_gen_rate=0.3,crossover_rate=0.2,
    #                                         mutation_rate_2=0.99,max_length=4,max_loop_depth=3,qubit_limit=10,
    #                                     compare_method='combined',pop_size=50, generations=400, rep=3)

    # # Run experiments
    # all_best_scores, all_scores, all_individual = plotter.run_experiments()
    # plotter.save_results('ry_experiments', all_best_scores, all_scores, all_individual)
    # all_best_scores, all_scores, all_individual = plotter.load_data('ry_experiments')
    # # Plot scores and save the figure
    # plotter.plot_scores(all_best_scores, save_path='Figure/ry_experiment.png')




    algorithm = ['qft_decom', 'qpe_dec', 'ghz_state']
    plotter = GeneticDecompilationPlotter(algorithm_name='example', algorithms=algorithm, mutation_rate=0.3,
                                        new_gen_rate=0.3,crossover_rate=0.2,
                                            mutation_rate_2=0.99,max_length=4,max_loop_depth=2,qubit_limit=10,
                                        compare_method='combined',pop_size=40, generations=500, rep=3)

    # Run experiments
    # all_best_scores, all_scores, all_individual = plotter.run_experiments()
    # plotter.save_results('advanced_algo', all_best_scores, all_scores, all_individual)
    all_best_scores, all_scores, all_individual = plotter.load_data('advanced_algo')
    # Plot scores and save the figure
    plotter.plot_scores(all_best_scores, save_path='Figure/advanced_algo.png')

    
    # algorithm = ['ry_c', 'ry_decomposed']
    # plotter = GeneticDecompilationPlotter(algorithm_name='example', algorithms=algorithm, mutation_rate=0.4,mutation_rate_2=0.99,
    #                                     new_gen_rate=0.2,crossover_rate=0.3,compare_method='combined',pop_size=40, generations=1000, rep=3)

    # # Run experiments
    # all_best_scores, all_scores, all_individual = plotter.run_experiments()
    # plotter.save_results('ry_experiments_2', all_best_scores, all_scores, all_individual)
    # all_best_scores, all_scores, all_individual = plotter.load_data('ry_experiments_2')
    # # Plot scores and save the figure
    # plotter.plot_scores(all_best_scores, save_path='Figure/ry_experiments_plot.png')


    # algorithm = ['qft_decom']
    # plotter = GeneticDecompilationPlotter(algorithm_name='example', algorithms=algorithm, mutation_rate=0.4,mutation_rate_2=0.99,
    #                                     new_gen_rate=0.2,crossover_rate=0.3,compare_method='combined',pop_size=40, generations=1000, rep=3)

    # # Run experiments
    # all_best_scores, all_scores, all_individual = plotter.run_experiments()
    # plotter.save_results('qft', all_best_scores, all_scores, all_individual)
    # all_best_scores, all_scores, all_individual = plotter.load_data('qft')
    # # Plot scores and save the figure
    # plotter.plot_scores(all_best_scores, save_path='Figure/qft.png')

    #####