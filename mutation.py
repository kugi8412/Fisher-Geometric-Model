import torch


def mutate_population(pop,
                      mutation_rate: float,
                      gene_mutation_rate: float,
                      mutation_strength: float) -> torch.Tensor:
    pop.genotypes = mutate_genotypes(pop.genotypes, mutation_rate, gene_mutation_rate, mutation_strength)


def mutate_genotypes(genotype: torch.Tensor,
                     mutation_rate: float,
                     gene_mutation_rate: float,
                     mutation_strength: float) -> torch.Tensor:
    """ Mutates only phenotypic genes (:,:-1,:).
    Adds noise result is truncated to [0,1].
    """
    mutation = torch.distributions.Normal(loc=torch.tensor([0.0], device=genotype.device), 
                                        scale=torch.sqrt(torch.tensor([mutation_strength],
                                        device=genotype.device))).sample(genotype.shape).squeeze(-1)
    mutation[:, -1, :] = 0.0 # not mutate the sex

    # if organism is mutating (n_org, n_genes, 2)
    mutate_ind = (torch.rand(genotype.shape[0]) < mutation_rate).unsqueeze(1).unsqueeze(2)\
                                                                .expand(-1, genotype.shape[1], 2)

    # if gene is mutating (n_org, n_genes, 2)
    mutate_gene = torch.rand(genotype.shape) < gene_mutation_rate

    # Update of genotypes of population in environment
    return (genotype + mutation * torch.logical_and(mutate_ind, mutate_gene).float().to(genotype.device)).clamp(0,1)
