import torch
from individual import Individual

class Population:
    def __init__(self, size, n_genes, device, area_width, area_height):
        self.size = size
        self.n_genes = n_genes  # np. 11: 10 fenotypowych + 1 płciowy
        self.device = device
        self.area_width = area_width
        self.area_height = area_height
        self.individuals = []
        for _ in range(size):
            # Genotyp: losujemy wartości z przedziału [0,1] dla genów fenotypowych
            genotype = torch.rand(n_genes, 2, device=device)
            # Ustalamy gen płci (ostatni) jako 0 lub 1 – oba allele mają tą samą wartość
            sex_val = float(torch.randint(0, 2, (1,), device=device).item())
            genotype[-1] = torch.tensor([sex_val, sex_val], device=device, dtype=torch.float)
            pos = torch.rand(2, device=device) * torch.tensor([area_width, area_height], device=device)
            self.individuals.append(Individual(genotype, pos))
    
    def get_phenotypes(self):
        """
        Oblicza fenotyp każdego osobnika i zwraca tensor (N, 2).
        """
        phenos = []
        for ind in self.individuals:
            pheno = torch.tensor(ind.get_phenotype(), device=self.device)
            phenos.append(pheno.unsqueeze(0))
        return torch.cat(phenos, dim=0)
    
    def remove_individuals(self, indices):
        self.individuals = [ind for i, ind in enumerate(self.individuals) if i not in indices]
        self.size = len(self.individuals)
    
    def update_positions(self, displacement):
        """
        Aktualizuje pozycje każdego osobnika.
        displacement: tensor (N, 2)
        """
        for i, ind in enumerate(self.individuals):
            new_pos = (ind.position + displacement[i]) % torch.tensor([self.area_width, self.area_height], device=self.device)
            ind.position = new_pos
