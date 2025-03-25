import torch
from typing import Dict
from population import Population

class FisherEnvironment:
    def __init__(self,
                 params: Dict[str, float],
                 device: str):
        """ Klasa przechowuje populację oraz
        aktualny, optymalny fenotyp.
        """
        self.params = params
        self.device = device
        self.c = torch.tensor(params['opt_drift'], device=device).expand(1, params['n_genes']-1, 2)
        
        self.pop = Population(
            params['n_organisms'],
            params['n_genes'],
            params['area_width'],
            params['area_height'],
            params['phenotype_matrix'],
            device
        )
        
        # Optymalny genotyp dla danego środowiska, bez znaczenia na płeć
        self.optimal_genotype = torch.rand((1, params['n_genes']-1, 2), device=device)
        
    def get_optimal_phenotype(self):
        """ Metoda oblicza optymalny fenotyp z genotypu
        """
        return torch.matmul(self.optimal_genotype.mean(-1), self.params['phenotype_matrix'].transpose(0,1))
    
    def calculate_fitness(self):
        """ Metoda służąca obliczeniu prawdopodobieństwa przeżycia
        każdego osobnika znajdującego się w populacji.
        """
        phenos = self.pop.get_phenotypes()
        optimal_pheno = self.get_optimal_phenotype()
        
        # Wyliczenia prawdopodobieństwa przeżycia z rozkładu normalnego
        phenotype_dist = torch.norm(phenos - optimal_pheno, dim=1)
        phenotype_fitness = torch.exp(-phenotype_dist**2 /
                                      (2 * (self.params['selection']**2)))
        
        return phenotype_fitness
    
    def update_optimal(self):
        """ Metoda zapewniająca mutację
        optymalnego genotypu, w celu zmiany środowiska
        """
        mutation = torch.normal(mean=self.c,
                                std=self.params['opt_noise'] * torch.ones_like(self.c))
        self.optimal_genotype = (self.optimal_genotype + mutation).clamp(0,1)
