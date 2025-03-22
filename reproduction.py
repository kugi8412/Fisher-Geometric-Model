import torch
from individual import Individual
from scipy.spatial import KDTree

def sexual_reproduction(population, max_size):
    """
    Rozmnażanie płciowe: każdy osobnik szuka partnera w pobliżu za pomocą KDTree.

    - Samce i samice są rozdzielone i zapisane w KDTree.
    - Dla każdego samca wybieramy najbliższą samicę.
    - Tworzymy nowego osobnika metodą crossing-over.
    """
    males = [ind for ind in population.individuals if ind.sex == 1]
    females = [ind for ind in population.individuals if ind.sex == 0]

    if not males or not females:
        return

    male_positions = torch.stack([m.position for m in males]).cpu().numpy()
    female_positions = torch.stack([f.position for f in females]).cpu().numpy()

    kdtree = KDTree(female_positions)  # Gogolewski KDETree

    new_individuals = []
    for male in males:
        dist, index = kdtree.query(male_positions)
        female = females[index]

        # Tworzymy potomka metodą crossing-over
        offspring_genotype = torch.zeros_like(male.genotype)
        for gene_idx in range(population.n_genes):
            pick_parent1 = torch.randint(0, 2, (1,)).item()
            pick_parent2 = torch.randint(0, 2, (1,)).item()
            offspring_genotype[gene_idx, 0] = male.genotype[gene_idx, pick_parent1]
            offspring_genotype[gene_idx, 1] = female.genotype[gene_idx, pick_parent2]

        # Losowanie płci
        offspring_sex = torch.randint(0, 2, (1,)).item()
        offspring_position = (male.position + female.position) / 2.0

        new_individuals.append(Individual(offspring_genotype, offspring_position, offspring_sex))

    # Dodajemy nowych osobników ograniczając ich liczbę
    population.individuals += new_individuals
    population.individuals = population.individuals[:max_size]
    population.size = len(population.individuals)
