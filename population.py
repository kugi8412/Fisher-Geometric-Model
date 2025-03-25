import torch

class Population:
    def __init__(self,
                 size: int,
                 n_genes: int,
                 area_width: int,
                 area_height: int,
                 genes_to_phenos: torch.tensor,
                 device: str):
        """ Klasa przechowująca wszystkie
        osobniki w jednej macierzy.
        """
        self.size = size
        self.n_genes = n_genes
        self.area_width = area_width
        self.area_height = area_height
        self.genos_to_phenos = genes_to_phenos
        self.device = device
        
        # Inicjalizacja genotypu osobników z ostanim genem determinującym płeć
        self.genotypes = torch.rand(size, n_genes-1, 2, device=device)
        last_gene_values = torch.randint(0, 2, (size, 1), device=device)
        last_gene = last_gene_values.unsqueeze(2).expand(-1, -1, 2)
        self.genotypes = torch.cat([self.genotypes, last_gene], dim=1)
        
        # Inicjalizacja początkowej położenia osobników
        self.positions = torch.rand(size, 2, device=device) * torch.tensor(
            [area_width, area_height], device=device)
    
    def get_phenotypes(self):
        """ Metoda oblicza fenotyp z uwzględnieniem wag genów
        """
        return torch.matmul(self.genotypes[:,:-1].mean(-1), self.genos_to_phenos)
    
    def update_positions(self, displacement: torch.tensor):
        """ Metoda aktualizuje położenie osobników.
        Zakłądamy, że plansza jest torusem w R^2.
        """
        self.positions = (self.positions + displacement) % torch.tensor(
            [self.area_width, self.area_height], device=self.device)

    def get_sex_mask(self):
        """ Metoda zwraca maskę,
        gdzie True to kobieta, a False to Mężczyzna.
        (Zero to kobieta, jedynka to mężczyzna)
        """
        return self.genotypes[:, -1, 0] < 0.5
