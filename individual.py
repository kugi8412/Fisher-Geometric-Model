import torch

class Individual:
    def __init__(self, genotype, position, sex):
        """
        Klasa przechowująca dane osobnika.
        
        :param genotype: tensor (n_genes, 2) - każdy gen ma dwa allele
        :param position: tensor (2,) - współrzędne x, y
        :param sex: int (0 - samica, 1 - samiec)
        """
        self.genotype = genotype  # Genotyp (dwualleliczny)
        self.position = position  # Pozycja (x, y)
        self.sex = sex  # Płeć (0 - samica, 1 - samiec)

    def get_phenotype(self):
        """
        Oblicza fenotyp jako średnią wartości alleli w każdym genie.
        """
        return self.genotype.mean(dim=-1)

    def move(self, speed):
        """
        Porusza osobnika o losowy wektor w przestrzeni (torus).
        """
        angle = torch.tensor(torch.rand(1).item() * 2 * torch.pi)
        delta = speed * torch.tensor([torch.cos(angle), torch.sin(angle)]).to('cuda')
        self.position = (self.position + delta) % 10  # Torus o wymiarach [0,10] x [0,10]
