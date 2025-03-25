import torch
from sklearn.neighbors import KDTree
import streamlit as st

def reproduce(env):
    pop = env.pop
    device = pop.device
    
    female_mask = pop.get_sex_mask()
    male_mask = ~female_mask
    
    if female_mask.sum() == 0 or male_mask.sum() == 0:
        st.warning("Brak osobników jednej z płci")
        return

    # Konwersja do CPU dla KDTree
    females_pos = pop.positions[female_mask].cpu().numpy()
    males_pos = pop.positions[male_mask].cpu().numpy()
    
    tree = KDTree(males_pos)
    distances, neighbors = tree.query(females_pos, return_distance=True)
    distances = torch.tensor(distances.squeeze(), device=device)
    neighbors = torch.tensor(neighbors.squeeze(), device=device)

    phenotypes = pop.get_phenotypes()
    radius = phenotypes[female_mask, 1]
    reproduction_prob = 1 - pop.size / (2 * env.params['n_organisms'])
    
    is_reproducing = (distances < radius) & (torch.rand_like(distances) < reproduction_prob)
    
    if is_reproducing.any():
        # Krzyżowanie genów
        female_genotypes = pop.genotypes[female_mask][is_reproducing]
        male_genotypes = pop.genotypes[male_mask][neighbors[is_reproducing]]
        
        mask = torch.rand_like(female_genotypes[:, :-1]) < 0.5
        new_genotypes = torch.where(mask, female_genotypes[:, :-1], male_genotypes[:, :-1])
        
        # Dodawanie nowych osobników
        new_sex = torch.randint(0, 2, (new_genotypes.shape[0], 1, 2), device=device)
        pop.genotypes = torch.cat([pop.genotypes, torch.cat([new_genotypes, new_sex], dim=1)])
        pop.positions = torch.cat([pop.positions, (pop.positions[female_mask][is_reproducing] + pop.positions[male_mask][neighbors[is_reproducing]])/2])
        pop.size = pop.genotypes.shape[0]
