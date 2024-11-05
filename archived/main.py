from decomplier import *

if __name__ == "__main__":
    decompiler = genetic_Decompiler(operations=['h', 'x', 'rx', 'ry', 'rz'],
                                generations=10,algorithm_name='rx_c',compare_method='seq_similarity',
                                perform_crossover=False,perform_mutation=True,pop_size=100,new_gen_rate=0.6)
    best_code, best_score, best_generation_index = decompiler.run()
    print(best_code, '\n', best_score, '\n',best_generation_index)