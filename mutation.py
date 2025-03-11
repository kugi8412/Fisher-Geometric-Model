# mutation.py

import numpy as np

def mutate_individual(individual, mu, mu_c, xi):
    """
    Mutacja osobnika: 
      - Z prawdopodobieństwem mu osobnik ulega mutacji
      - Każda cecha p_i mutuje niezależnie z prawdopodobieństwem mu_c
      - Zmiana mutacyjna jest losowana z N(0, xi^2)
    """
    if np.random.rand() < mu:
        phenotype = individual.get_phenotype().copy()
        for i in range(len(phenotype)):
            if np.random.rand() < mu_c:
                phenotype[i] += np.random.normal(0.0, xi)
        individual.set_phenotype(phenotype)

def mutate_population(population, mu, mu_c, xi):
    """
    Mutuje całą populację (lista osobników).
    """
    for ind in population.get_individuals():
        mutate_individual(ind, mu, mu_c, xi)
