import torch
from typing import Dict
from population import Population


class FisherEnvironment:
    def __init__(self,
                 params: Dict[str, float],
                 device: str):
        """ Class contains the population and
        the current optimal phenotype.
        """
        self.params = params
        self.device = device
        self.c = torch.tensor(params['opt_drift'], device=device, dtype=torch.float32)
        self.c = self.c.view(1, params['n_genes']-1, 2)
        self.current_step = 0
        self.pop = Population(
            params['n_organisms'],
            params['n_genes'],
            params['area_width'],
            params['area_height'],
            params['phenotype_matrix'],
            device
        )
        
        # Optimum genotype for a given environment, without regard to gender
        self.optimal_genotype = torch.rand((1, params['n_genes']-1, 2), 
                                         device=device, 
                                         dtype=torch.float32)
        
    def get_optimal_phenotype(self) -> torch.tensor:
        """ Method calculates the optimal phenotype from the genotype.
        """
        return torch.matmul(self.optimal_genotype.mean(-1), self.params['phenotype_matrix'])
    
    def calculate_fitness(self) -> torch.tensor:
        """ Method for calculating the probability of survival
        of each individual in a population.
        """
        phenos = self.pop.get_phenotypes()
        optimal_pheno = self.get_optimal_phenotype()
        
        #  Calculation of probability of survival from a normal distribution
        distances = torch.norm(phenos - optimal_pheno, dim=1)
        phenotype_fitness = torch.exp(-(distances**2) / (2 * (self.params['selection']**2)))
        
        return phenotype_fitness
    
    def update_optimal(self):
        """ Method to ensure mutation
        of the optimal genotype to change the environment.
        Each gene has a value from 0 to 1. Also recording
        in which generation the environment is.
        """
        mutation = torch.normal(
            mean=self.c,
            std=self.params['opt_noise'] * torch.ones_like(self.c)
        )
        self.optimal_genotype = (self.optimal_genotype + mutation).clamp(0, 1).float()
        self.current_step += 1
