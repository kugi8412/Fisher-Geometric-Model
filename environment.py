import torch
from population import Population

class FisherEnvironment:
    def __init__(self, params, device):
        self.params = params
        self.device = device
        self.area_width = params['area_width']
        self.area_height = params['area_height']
        self.pop = Population(params['n_organisms'], params['n_genes'], device, self.area_width, self.area_height)
        # Początkowe optimum fenotypowe – losujemy z przedziału [0, maksymalna wartość] dla każdej osi.
        # Maksymalna wartość dla jednej osi przy 10 genach fenotypowych wynosi 5.
        self.optimal = torch.rand(2, device=device) * ((params['n_genes'] - 1)/2)
    
    def update_optimal(self):
        # Optimum porusza się zgodnie z rozkładem normalnym – parametry podawane w params
        drift = torch.normal(self.params['opt_drift'], self.params['opt_noise'], size=(2,), device=self.device)
        self.optimal = (self.optimal + drift).clamp(0, (self.params['n_genes'] - 1)/2)
        # Przechowujemy optimum również w obiekcie pop, aby wykres fenotypowy mógł go wykorzystać
        self.pop.optimum = self.optimal.detach().cpu().numpy()
    
    def calculate_fitness(self):
        phenos = self.pop.get_phenotypes()
        distances = torch.norm(phenos - self.optimal, dim=1)**2
        fitness = torch.exp(- distances**2 / (2 * self.params['selection']**2))
        return fitness
