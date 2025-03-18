import torch

class Individual:
    def __init__(self, genotype, position, life=1):
        # genotype: tensor (n_genes, 2)
        self.genotype = genotype  
        self.position = position    # pozycja w przestrzeni fizycznej
        self.life = life
        # Płeć ustalana na podstawie ostatniego genu – średnia < 0.5 -> 0, inaczej 1
        self.sex = 0 if self.genotype[-1].mean().item() < 0.5 else 1

    def get_phenotype(self):
        """
        Oblicza fenotyp z genotypu:
         - Używamy tylko genów fenotypowych (indeksy 0..n_genes-2).
         - Dzielimy je na dwie grupy równej wielkości (dla 10 genów: 5 na oś X, 5 na oś Y).
         - Dla każdej grupy obliczamy sumę średnich wartości - maksymalnie 5.
         Zwracamy wynik jako tablicę numpy o kształcie (2,).
        """
        # Obliczamy średnie dla każdego genu (dla każdego allele)
        gene_means = self.genotype[:-1].mean(dim=-1)  # tensor kształtu (n_genes-1,)
        n = gene_means.shape[0]  # np. 10
        half = n // 2
        x = gene_means[:half].sum()
        y = gene_means[half:].sum()
        phenotype = torch.tensor([x, y])
        # Ograniczamy fenotyp do przedziału [0, half] (czyli [0,5] przy 10 genach)
        phenotype = torch.clamp(phenotype, 0, half)
        return phenotype.cpu().numpy()
