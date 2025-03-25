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
        self.phenotype_matrix = genes_to_phenos
        self.device = device
        
        # Inicjalizacja genotypu osobników z ostanim genem determinującym płeć
        self.genotypes = torch.cat([torch.rand(size=(size, n_genes-1, 2), device=device),
                                    torch.randint(0, 2, size=(size, 1, 1), device=device).expand(-1,-1,2)],dim=1)
        
        # Inicjalizacja początkowej położenia osobników
        self.positions = torch.rand(size, 2, device=device) * torch.tensor(
                                    [area_width, area_height], device=device)
    
    def get_phenotypes(self):
        """
        Oblicza fenotyp każdego osobnika i zwraca tensor (N, 2),
        za wypadkowy genotyp uznajemy średnią z obu alleli.
        """
        return torch.matmul(self.genotypes.mean(-1)[:,:-1], self.phenotype_matrix.transpose(0,1))
    
    def update_positions(self, displacement):
        """
        Aktualizuje pozycje każdego osobnika.
        displacement: tensor (N, 2)
        """
        self.positions = (self.positions + displacement) % torch.tensor(
                        [self.area_width, self.area_height], device=self.device)

    def get_sex_mask(self):
        """ Metoda zwraca maskę,
        gdzie True to kobieta, a False to Mężczyzna.
        (Zero to kobieta, jedynka to mężczyzna)
        """
        return self.genotypes[:, -1, 0] < 0.5
