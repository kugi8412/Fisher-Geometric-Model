import numpy as np

def fitness_function(phenotype, alpha, sigma):
    """
    Oblicza fitness jako exp(-||p - alpha||^2/(2*sigma^2)).
    phenotype i alpha to tablice numpy o kształcie (2,).
    """
    diff = phenotype - alpha
    dist_sq = np.sum(diff**2)
    return np.exp(-dist_sq / (2 * sigma**2))

def apply_fitness_selection(population, alpha, sigma):
    """
    Dla każdego osobnika oblicza fitness. Z prawdopodobieństwem równym (1 - fitness)
    osobnik traci jedno życie. Jeśli jego life spadnie do 0, zostaje usunięty.
    """
    survivors = []
    for ind in population.individuals:
        f = fitness_function(ind.get_phenotype(), alpha, sigma)
        if np.random.rand() < (1 - f):
            ind.life -= 1
        if ind.life > 0:
            survivors.append(ind)
    return survivors
