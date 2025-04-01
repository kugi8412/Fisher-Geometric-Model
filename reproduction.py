import torch
from sklearn.neighbors import KDTree


def reproduce(env) -> bool:
    """ Function responsible for the reproduction
    of individuals that survive selection.
    Using KDTree for faster computation.
    If still remaining any individuals
    returns False and continue the simulation.
    """
    pop = env.pop
    device = pop.device

    female_mask = pop.get_sex_mask()
    male_mask = ~female_mask
    
    if female_mask.sum() == 0 or male_mask.sum() == 0:
        return True

    # Conversion to numpy for KDTree
    females_pos = pop.positions[female_mask].cpu().numpy()
    males_pos = pop.positions[male_mask].cpu().numpy()
    tree = KDTree(males_pos)
    distances, neighbors = tree.query(females_pos, k=1)
    distances = torch.tensor(distances.squeeze(), device=device)
    neighbors = torch.tensor(neighbors.squeeze(), device=device)
    
    # Initialisation of conditions
    phenotypes = pop.get_phenotypes()
    radius = phenotypes[female_mask, 1] * env.params['radius_multiplier']
    conditions = torch.ones_like(distances, dtype=torch.bool)
    reproduction_prob = torch.ones_like(distances)

    # Probability of reproduction is fitness
    if 'fitness' in env.params['reproduction_factors']:
        fitness = env.calculate_fitness()[female_mask]
        try:
            reproduction_prob *= fitness
        except RuntimeError:
            reproduction_prob = torch.zeros_like(distances)

    # Fitness Threshold
    if 'fitness_threshold' in env.params['reproduction_factors']:
        fitness = env.calculate_fitness()[female_mask]
        try:
            conditions &= (fitness > env.params['min_fitness'])
        except RuntimeError:
            conditions = torch.zeros_like(distances)

    # Environmnet capacity (logistic curve)
    if 'capacity' in env.params['reproduction_factors']:
        capacity_factor = 1 - (pop.size / env.params['max_population'])
        reproduction_prob *= capacity_factor

    # Probability of Reproduction
    is_reproducing = (distances < radius) & conditions & (torch.rand_like(distances) < reproduction_prob)
    
    # Crossing-over
    if is_reproducing.any():
        female_genotypes = pop.genotypes[female_mask]
        females_pos = pop.positions[female_mask]
        male_genotypes = pop.genotypes[male_mask][neighbors]
        males_pos = pop.positions[male_mask][neighbors]
        
        # Addition of a dimension deleted at indexing
        if len(female_genotypes.shape) == 2:
            female_genotypes = female_genotypes.unsqueeze(0)
            females_pos = females_pos.unsqueeze(0)
        if len(male_genotypes.shape) == 2:
            male_genotypes = male_genotypes.unsqueeze(0)
            males_pos = males_pos.unsqueeze(0)
        
        female_genotypes = female_genotypes[is_reproducing]
        females_pos = females_pos[is_reproducing]
        male_genotypes = male_genotypes[is_reproducing]
        males_pos = males_pos[is_reproducing]
        
        mask = torch.rand_like(female_genotypes[:, :-1]) < 0.5
        new_genotypes = torch.where(mask, female_genotypes[:, :-1], male_genotypes[:, :-1])
        
        # Adding new individuals
        new_sex = torch.randint(0, 2, (new_genotypes.shape[0], 1, 2), device=device)
        pop.genotypes = torch.cat([pop.genotypes, torch.cat([new_genotypes, new_sex], dim=1)])
        pop.positions = torch.cat([pop.positions, (females_pos + males_pos)/2])
        pop.size = pop.genotypes.shape[0]

        # Maximal size of population
        if pop.size > env.params['max_population']:
            if env.params.get('selection_type') == "fitness":
                fitness = env.calculate_fitness()
                _, indices = torch.topk(fitness, env.params['max_population'])
            else:
                indices = torch.randperm(pop.size)[:env.params['max_population']]
            
            pop.genotypes = pop.genotypes[indices]
            pop.positions = pop.positions[indices]
            pop.size = env.params['max_population']

    return False
