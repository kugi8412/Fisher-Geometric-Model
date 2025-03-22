import torch
from individual import Individual

class Population:
    def __init__(self, size, n_genes, device):
        """
        Inicjalizuje populację losowymi osobnikami.

        :param size: liczba osobników
        :param n_genes: liczba genów
        """
        self.size = size
        self.n_genes = n_genes
        self.device = device
        # CHANGE, potrzebuję położenia osobników, 
        # self.individuals = torch.randn((size, n_genes)).abs()

        self.individuals = []
        for _ in range(size):
            genotype = torch.rand(n_genes, 2, device=device)  # Genotyp (n_genes, 2)
            position = torch.rand(2, device=device) * 10.0  # Pozycja (x, y)
            sex = torch.randint(0, 2, (1,)).item()  # Płeć (0 - samica, 1 - samiec)
            self.individuals.append(Individual(genotype, position, sex))

    def move_population(self, speed):
        """
        Przesuwa całą populację o losowe wartości w przestrzeni.
        """
        for ind in self.individuals:
            ind.move(speed)

    def get_positions(self):
        """
        Zwraca tensor pozycji wszystkich osobników.
        """
        return torch.stack([ind.position for ind in self.individuals])

    def get_individuals(self):
        return self.individuals

    def set_individuals(self, new_individuals):
        self.individuals = new_individuals
