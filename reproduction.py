import torch
from individual import Individual
import random

def reproduce(env):
    """
    Reprodukcja odbywa się między osobnikami różnych płci.
    Dla każdej "kobiety" (przyjmujemy, że sex == 0) wybieramy mężczyznę (sex == 1)
    na podstawie odległości w przestrzeni fizycznej (z uwzględnieniem torusa).
    Następnie tworzymy potomka metodą crossing-over: dla każdego genu fenotypowego
    (indeksy 0..n_genes-2) losowo wybieramy allele od mamy i taty.
    """
    pop = env.pop
    new_individuals = []
    phenos = pop.get_phenotypes()
    
    for i, ind in enumerate(pop.individuals):
        if ind.sex == 0:  # traktujemy jako "kobieta"
            male_indices = [j for j, other in enumerate(pop.individuals) if other.sex == 1]
            if not male_indices:
                continue
            pos_f = ind.position
            distances = []
            for j in male_indices:
                pos_m = pop.individuals[j].position
                diff = pos_f - pos_m
                diff = (diff + pop.area_width/2) % pop.area_width - pop.area_width/2
                distances.append(torch.norm(diff))
            distances = torch.tensor(distances, device=pop.device)
            # Używamy drugiej składowej fenotypu "kobiety" jako promienia rozmnażania
            r = phenos[i, 1]
            prob = torch.exp(- distances**2 / (2 * (r**2 + 1e-6)))
            if prob.numel() == 0 or prob.max().item() < torch.rand(1).item():
                continue
            weights = prob / prob.sum()
            chosen = torch.multinomial(weights, 1).item()
            male_index = male_indices[chosen]
            male = pop.individuals[male_index]
            offspring_genotype = torch.zeros(pop.n_genes, 2, device=pop.device)
            for j in range(0, pop.n_genes - 1):
                pick_f = torch.randint(0, 2, (1,)).item()
                pick_m = torch.randint(0, 2, (1,)).item()
                offspring_genotype[j, 0] = ind.genotype[j, pick_f]
                offspring_genotype[j, 1] = male.genotype[j, pick_m]
            offspring_sex = float(torch.randint(0, 2, (1,)).item())
            offspring_genotype[-1] = torch.tensor([offspring_sex, offspring_sex], device=pop.device, dtype=torch.float)
            offspring_position = (ind.position + male.position) / 2.0 + torch.randn(2, device=pop.device)*0.1
            new_individuals.append(Individual(offspring_genotype, offspring_position))
    total_needed = pop.size
    if len(new_individuals) < total_needed:
        additional = [random.choice(pop.individuals) for _ in range(total_needed - len(new_individuals))]
        new_individuals.extend(additional)
    pop.individuals = new_individuals
    pop.size = len(new_individuals)
