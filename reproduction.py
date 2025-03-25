import torch


def reproduce(env):
    pop = env.population
    device = pop.device
    
    # Poprawna maska płci
    female_mask = pop.genotypes[:, -1, 0] < 0.5
    male_mask = ~female_mask
    
    if female_mask.sum() == 0 or male_mask.sum() == 0:
        return

    # Efektywne obliczanie odległości
    females_pos = pop.positions[female_mask]
    males_pos = pop.positions[male_mask]
    dists = torch.cdist(females_pos, males_pos)
    
    # Warunki reprodukcji
    min_dists, male_indices = torch.min(dists, dim=1)
    phenotypes = pop.get_phenotypes()
    reproduction_prob = 1 - pop.size / (2 * env.params['n_organisms'])
    
    is_reproducing = (
        (min_dists < phenotypes[female_mask, 1]) &
        (torch.rand(len(females_pos), device=device) < reproduction_prob)
    )
    
    # Krzyżowanie genów
    if is_reproducing.any():
        female_genotypes = pop.genotypes[female_mask][is_reproducing]
        male_genotypes = pop.genotypes[male_mask][male_indices[is_reproducing]]
        
        # Losowy wybór alleli
        mask = torch.rand_like(female_genotypes[:, :-1]) < 0.5
        new_genotypes = torch.where(mask, female_genotypes[:, :-1], male_genotypes[:, :-1])
        
        # Dodawanie nowych osobników
        pop.genotypes = torch.cat([
            pop.genotypes,
            torch.cat([new_genotypes, torch.randint(0, 2, (len(new_genotypes), 1, 2), device=device)], dim=1)
        ], dim=0)
        
        pop.size = pop.genotypes.shape[0]
