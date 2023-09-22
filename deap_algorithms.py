import numpy as np
def local_search_1(X, i):
    p = np.random.rand()
    if p < 0.2:
        X[i] =np.random.rand()
    return np.array(X)


def local_search_2(X, i):
    X_prime = []
    p = np.random.rand()
    for n in X:
        if p < 0.05:
            n += np.random.rand()
        X_prime.append(n)
        p = np.random.rand()
    return np.array(X_prime)
def simmulated_annealing(X, toolbox, k_max=100, nn=local_search_2):
    def P(e, e_prime, T):
        return np.exp(-(e_prime-e)/T)

    x = X 
    best_opt = (None, 0)
    for k in range(k_max-1):
        new_x = nn(x, int(np.random.randint(len(x))))
        new_x_fitness = toolbox.evaluate(new_x)[0]
        best_opt = (new_x, new_x_fitness) if best_opt[1] < new_x_fitness else best_opt
        #print(f'Eval SA: {np.sum(np.abs(x-new_x))}')
        if new_x_fitness  >= X.fitness.values[0]:
            x = new_x
            continue
        T = 1 - ((k+1)/k_max)
        if P(X.fitness.values[0] / 100, new_x_fitness/100, T) >= np.random.rand():
            x = new_x
    print(f'Best candidate: {best_opt[1]}')
    return x
