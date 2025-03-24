import torch

def reproduce(env):
    pop = env.population
    device = pop.device

    # Get masks for females and males
    female_mask = pop.get_sex_mask()
    male_mask = ~female_mask  # Opposite of female mask
    
    if female_mask.sum() == 0 or male_mask.sum() == 0:
        return  # No reproduction possible

    females_pos = pop.positions[female_mask]
    males_pos = pop.positions[male_mask]

    # Compute pairwise Euclidean distances using cdist
    dists = torch.cdist(females_pos, males_pos)

    # Find nearest male for each female
    min_dists, male_indices = torch.min(dists, dim=1)

    # Compute reproduction probabilities
    phenotypes = pop.get_phenotypes(env.params['phenotype_matrix'])
    female_radii = phenotypes[female_mask, 1]  # Reproduction range
    max_population = 2 * env.params['n_organisms']
    reproduction_prob = 1 - pop.size / max_population

    is_reproducing = (min_dists < female_radii) & (torch.rand(len(min_dists), device=device) < reproduction_prob)

    if is_reproducing.any():
        # Crossing over: Randomly take alleles from parents
        female_genotypes = pop.genotypes[female_mask][is_reproducing]
        male_genotypes = pop.genotypes[male_mask][male_indices[is_reproducing]]

        take_f_allele = torch.rand((is_reproducing.sum(), pop.n_genes-1, 2), device=device) < 0.5
        new_genotypes = torch.where(take_f_allele, female_genotypes[:, :-1, :], male_genotypes[:, :-1, :])

        # Set last gene (sex)
        new_sex_values = torch.randint(0, 2, (is_reproducing.sum(), 1), device=device)
        new_sex = new_sex_values.unsqueeze(2).expand(-1, -1, 2)
        new_genotypes = torch.cat([new_genotypes, new_sex], dim=1)

        # Compute new positions (midpoint of parents)
        new_positions = (pop.positions[female_mask][is_reproducing] + pop.positions[male_mask][male_indices[is_reproducing]]) / 2

        # Update population
        pop.genotypes = torch.cat([pop.genotypes, new_genotypes], dim=0)
        pop.positions = torch.cat([pop.positions, new_positions], dim=0)
        pop.size = pop.genotypes.shape[0]
