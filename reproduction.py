import torch
from sklearn.neighbors import KDTree
import streamlit as st

def reproduce(env):
    pop = env.pop
    device = pop.device
    
    female_mask = pop.get_sex_mask()
    male_mask = ~female_mask
    
    if female_mask.sum() == 0 or male_mask.sum() == 0:
        return True

    # Konwersja do CPU dla KDTree
    females_pos = pop.positions[female_mask].cpu().numpy()
    males_pos = pop.positions[male_mask].cpu().numpy()
    
    tree = KDTree(males_pos)
    distances, neighbors = tree.query(females_pos, return_distance=True)
    distances = torch.tensor(distances.squeeze(), device=device)
    neighbors = torch.tensor(neighbors.squeeze(), device=device)

    phenotypes = pop.get_phenotypes()
    radius = phenotypes[female_mask, 1] * env.params['radius_multiplier']
    reproduction_prob = 1 - pop.size / (env.params['max_pop_size'])
    
    is_reproducing = (distances < radius) & (torch.rand_like(distances) < reproduction_prob)
    
    # Krzyżowanie genów
    if is_reproducing.any():
        female_genotypes = pop.genotypes[female_mask]
        females_pos = pop.positions[female_mask]
        male_genotypes = pop.genotypes[male_mask][neighbors]
        males_pos = pop.positions[male_mask][neighbors]
        
        # Dodanie wymiaru singletonowego usuniętego przy indeksowaniu
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
        
        # Dodawanie nowych osobników
        new_sex = torch.randint(0, 2, (new_genotypes.shape[0], 1, 2), device=device)
        pop.genotypes = torch.cat([pop.genotypes, torch.cat([new_genotypes, new_sex], dim=1)])
        pop.positions = torch.cat([pop.positions, (females_pos + males_pos)/2])
        pop.size = pop.genotypes.shape[0]

    return False
