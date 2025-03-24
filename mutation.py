# mutation.py

import torch

def mutate_population(pop, mutation_rate, gene_mutation_rate, mutation_strength):
    pop.genotypes = mutate_genotypes(pop.genotypes, mutation_rate, gene_mutation_rate, mutation_strength)

def mutate_genotypes(genotype, mutation_rate, gene_mutation_rate, mutation_strength):
    """
    Mutuje tylko geny fenotypowe (:,:-1,:).
    Dodaje szum; wynik jest obcinany do [0,1].
    """
    mutation = torch.distributions.Normal(loc=torch.tensor([0.0], device=genotype.device), 
                                          scale=torch.tensor([mutation_strength], device=genotype.device)).sample(genotype.shape).squeeze(-1)
    mutation[:,-1,:] = 0.0 # nie mutujemy płci
    mutate_ind = (torch.rand(genotype.shape[0]) < mutation_rate).unsqueeze(1).unsqueeze(2).expand(-1,genotype.shape[1],2) # (n_org, n_genes, 2), czy dany organizm mutuje
    mutate_gene = torch.rand(genotype.shape) < gene_mutation_rate # (n_org, n_genes, 2), czy dany gen mutuje
    return (genotype + mutation * torch.logical_and(mutate_ind, mutate_gene).float()).clamp(0,1)
