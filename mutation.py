import torch

def mutate_population(pop, mutation_rate, mutation_strength):
    """
    Mutuje tylko geny fenotypowe (indeksy 0..n_genes-2).
    Dodaje szum; wynik "zawija" się modulo 1, bo wartości genów są w [0,1].
    """
    for ind in pop.individuals:
        for j in range(0, pop.n_genes - 1):
            for allele in range(2):
                if torch.rand(1).item() < mutation_rate:
                    noise = torch.randn(1).item() * mutation_strength
                    new_val = ind.genotype[j, allele].item() + noise
                    ind.genotype[j, allele] = new_val % 1.0
