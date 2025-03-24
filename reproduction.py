import torch
from sklearn.neighbors import KDTree
import streamlit as st
from mutation import mutate_genotypes

def reproduce(env):
    """
    Reprodukcja odbywa się między osobnikami różnych płci.
    Dla każdej "kobiety" (przyjmujemy, że sex == 0) wybieramy najbliższego mężczyznę 
    (sex == 1) w promienu zadanym przez fenotyp.
    Następnie tworzymy potomka metodą crossing-over: dla każdego genu fenotypowego
    (:,:-1,:) losowo wybieramy allele od mamy i taty.
    """
    genotypes = env.pop.genotypes
    pop = env.pop

    female_ind, male_ind = genotypes.mean(-1)[:,-1]==0, genotypes.mean(-1)[:,-1]==1
    if female_ind.sum() == 0 or male_ind.sum() == 0:
            st.warning("Zbyt mało osobników – symulacja zakończona.")
            st.stop()
            return
    females, males = pop.positions[female_ind], pop.positions[male_ind]

    # Stworzenie par i obliczenie odległości
    tree = KDTree(males) # Struktura przechowująca pozycje mężczyzn
    distances, neighbors = tree.query(females, return_distance=True) # Dla każdej kobiety zwraca indeks i odległość do najbliższej męzczyzny
    distances, neighbors = torch.tensor(distances.squeeze(1), device=genotypes.device), torch.tensor(neighbors.squeeze(1), device=genotypes.device)

    phenotypes = pop.get_phenotypes()
    radius = phenotypes[female_ind,1]

    # Jeśli drugi osobnik jest w promieniu zadanym przez fenotyp to rozmnażanie zachodzi z prawdopodobieństwem (1 - n_organisms_t/K)
    # gdzie K jest dwukrotnością początkowej liczby osobników, ma to na celu zapobiec wybuchowi liczebności populacji.

    is_reproducing = torch.logical_and(distances < radius, torch.rand_like(distances, device=genotypes.device) 
                                       < ((1-env.pop.size/env.params['n_organisms']*2)))

    if is_reproducing.sum() == 0:
        return

    # Dla każdego genu losowo wybierany jest allel od jednego z rodziców
    take_f_allele = torch.rand(is_reproducing.sum(), genotypes.shape[1], genotypes.shape[2], device=genotypes.device) < 0.5
    new_genotypes = torch.where(take_f_allele, genotypes[female_ind][is_reproducing], genotypes[male_ind][neighbors][is_reproducing])
    new_genotypes[:, -1, :] = torch.bernoulli(torch.ones(new_genotypes.shape[0], device=new_genotypes.device)*0.5).unsqueeze(1).expand(-1,2)

    # Pozycja potomka jest srednią z pozycji rodziców
    new_positions = (females[is_reproducing] + males[neighbors][is_reproducing])/2

    pop.positions = torch.cat([pop.positions, new_positions], dim=0)
    pop.genotypes = torch.cat([pop.genotypes, new_genotypes], dim=0)
    pop.size += is_reproducing.sum()