# population.py

import torch
from config import GENOTYPE_MATRIX

class Population:
    def __init__(self, size, n_genes, device, area_width, area_height):
        self.size = size
        self.n_genes = n_genes  # np. 11: 10 fenotypowych + 1 płciowy
        self.device = device
        self.area_width = area_width
        self.area_height = area_height
        self.genotypes = torch.cat([torch.rand(size=(size, n_genes-1, 2), device=device), # (n_org, n_genes, n_alleles)
                                      torch.randint(0, 2, size=(size, 1, 1), device=device).expand(-1,-1,2)],dim=1)
        self.positions = torch.rand(size=(size, 2), device=device) * torch.tensor([area_width, area_height], device=device)
        self.phenotype_matrix = GENOTYPE_MATRIX.T # (n_genes-1, n_alleles)
    
    def get_phenotypes(self):
        """
        Oblicza fenotyp każdego osobnika i zwraca tensor (N, 2),
        za wypadkowy genotyp uznajemy średnią z obu alleli.
        """
        return torch.matmul(self.genotypes.mean(-1)[:,:-1], self.phenotype_matrix)
    
    def update_positions(self, displacement):
        """
        Aktualizuje pozycje każdego osobnika.
        displacement: tensor (N, 2)
        """
        self.positions = (self.positions + displacement) % torch.tensor([self.area_width, self.area_height], device=self.device)
