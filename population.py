import torch


class Population:
    def __init__(self,
                 size: int,
                 n_genes: int,
                 area_width: int,
                 area_height: int,
                 genes_to_phenos: torch.tensor,
                 device: str):
        """ A class containing all individuals in a single matrix,
        where N is the number of genes that determine phenotype, not sex.
        """
        self.size = size
        self.n_genes = n_genes
        self.area_width = area_width
        self.area_height = area_height
        self.phenotype_matrix = genes_to_phenos
        self.device = device
        
        # Initialisation of the genotype of individuals with the last sex-determining gene
        self.genotypes = torch.cat([torch.rand(size=(size, n_genes-1, 2), device=device),
                                    torch.randint(0, 2, size=(size, 1, 1), device=device).expand(-1,-1,2)],dim=1)
        
        # Initialisation of the initial position of individuals
        self.positions = torch.rand(size, 2, device=device) *\
                        torch.tensor([area_width, area_height], device=device)
    
    def get_phenotypes(self) -> torch.tensor:
        """ Calculates the phenotype of each individual and
        returns a tensor of (N, 2). Consider the average of the two
        alleles as the resultant genotype.
        """
        return torch.matmul(self.genotypes.mean(-1)[:, :-1], self.phenotype_matrix)
    
    def update_positions(self,
                         displacement: torch.tensor) -> None:
        """ Updates the position of each individual
        in two-dimensional space.
        """
        self.positions = (self.positions + displacement) % torch.tensor(
                        [self.area_width, self.area_height], device=self.device)

    def get_sex_mask(self) -> torch.tensor:
        """ The method returns a mask,
        where True is Female and False is Male.
        (Zero is female, one is male).
        """
        return self.genotypes[:, -1, 0] < 0.5
