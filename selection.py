# selection.py

import torch

def fitness_function(phenotype, alpha, sigma) -> torch.Tensor:
    """
    Funkcja fitness: phi_alpha(p) = exp( -||p - alpha||^2 / (2*sigma^2) )
    :param phenotype: fenotyp osobnika (np.array)
    :param alpha: optymalny fenotyp (np.array)
    :param sigma: odchylenie (float) kontrolujące siłę selekcji
    """
    diff = phenotype - alpha
    dist_sq = torch.sum(diff**2, dim=1)
    return torch.exp(-dist_sq / (2 * sigma**2))

def selection(population, alpha, sigma, N, threshold):
    individuals = population.get_individuals()
    fitnesses = fitness_function(individuals, alpha, sigma)
    if fitnesses.sum() == 0:
        probabilities = torch.tensor([1/len(fitnesses)]*len(fitnesses))
    else:
        probabilities = fitnesses / fitnesses.sum()
    new_individuals = individuals[torch.multinomial(probabilities, num_samples=individuals.shape[0])]
    # aktualnie osobniki umierające w danym kroku mogą nadal się rozmnażać, można to zmienić ale nie wiem czy ma to istotne znaczenie
    individuals = torch.masked_scatter(individuals, fitnesses.unsqueeze(1).expand(-1,individuals.shape[1]) < threshold, new_individuals)
    return individuals
