import torch

def mutate_population(pop, mutation_rate, gene_mutation_rate, mutation_strength):
    genotypes = pop.genotypes
    device = genotypes.device
    
    # Create mutation tensor on correct device
    mutation = torch.normal(
        mean=0.0, 
        std=mutation_strength, 
        size=genotypes.shape, 
        device=device
    ).squeeze(-1)
    
    mutation[:,-1,:] = 0.0  # Don't mutate sex gene
    
    # Create masks on correct device
    mutate_ind = torch.rand(genotypes.shape[0], device=device) < mutation_rate
    mutate_ind = mutate_ind.unsqueeze(1).unsqueeze(2).expand_as(genotypes)
    
    mutate_gene = torch.rand(genotypes.shape, device=device) < gene_mutation_rate
    
    # Apply mutation
    pop.genotypes += mutation * torch.logical_and(mutate_ind, mutate_gene).float()
    pop.genotypes = pop.genotypes.clamp(0, 1)
