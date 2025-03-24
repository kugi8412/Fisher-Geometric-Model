import torch


def fitness_function(phenotype, alpha, sigma) -> torch.Tensor:
    """
    Funkcja fitness: phi_alpha(p) = exp( -||p - alpha||^2 / (2*sigma^2) )
    :param phenotype: tensor (N, 2) - fenotypy osobników
    :param alpha: tensor (2,) - optymalny fenotyp
    :param sigma: float - siła selekcji
    :return: tensor (N,) - wartości fitness dla każdego osobnika
    """
    diff = phenotype - alpha
    dist_sq = torch.sum(diff**2, dim=1)
    return torch.exp(-dist_sq / (2 * sigma**2))

def selection(population, alpha, sigma, max_population):
    """
    Selekcja naturalna:
    - Oblicza fitness każdego osobnika.
    - Osobniki z niskim fitness mają większe prawdopodobieństwo śmierci.
    - Tylko przetrwałe osobniki mogą rozmnażać się.
    
    :param population: obiekt klasy Population
    :param alpha: tensor (2,) - optymalny fenotyp
    :param sigma: float - siła selekcji
    :param max_population: int - maksymalna liczba osobników
    """
    individuals = population.individuals
    phenotypes = torch.stack([ind.get_phenotype() for ind in individuals])
    fitnesses = fitness_function(phenotypes, alpha, sigma)

    # Eliminacja osobników z prawdopodobieństwem 1 - fitness
    survival_mask = torch.rand(len(individuals)) < fitnesses
    survivors = [ind for i, ind in enumerate(individuals) if survival_mask[i]]

    # Ograniczenie liczby osobników do max_population
    if len(survivors) > max_population:
        probabilities = fitnesses[survival_mask] / fitnesses[survival_mask].sum()
        chosen_indices = torch.multinomial(probabilities, num_samples=max_population, replacement=False)
        survivors = [survivors[i] for i in chosen_indices]

    # Aktualizacja populacji - tylko ocalałe osobniki
    population.individuals = survivors
    population.size = len(survivors)

def apply_fitness_selection(env):
    """
    Usuwa osobniki z prawdopodobieństwem zależnym (1 - fitness)
    """
    fitnesses = env.calculate_fitness()
    device = fitnesses.device
    
    # Create survival mask on correct device
    survives = fitnesses > torch.rand(fitnesses.shape, device=device)
    
    # Apply mask
    env.population.genotypes = env.population.genotypes[survives]
    env.population.positions = env.population.positions[survives]
    env.population.size = env.population.genotypes.shape[0]
