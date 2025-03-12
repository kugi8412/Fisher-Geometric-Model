# mutation.py

import torch

def mutate_population(population, mu, mu_c, xi):
    """
    Mutuje całą populację.
    """
    individuals = population.get_individuals()
    mutation = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([xi])).sample(individuals.shape).squeeze(2)
    mutate_ind = (torch.rand(individuals.shape[0]) < mu).unsqueeze(1).expand(-1,individuals.shape[1])
    mutate_gene = torch.rand(individuals.shape) < mu_c
    individuals = torch.masked_scatter(individuals, torch.logical_and(mutate_ind, mutate_gene), individuals+mutation)
    population.set_individuals(individuals)

