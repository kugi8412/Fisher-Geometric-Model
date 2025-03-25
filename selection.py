import torch

def apply_fitness_selection(env):
    """
    Usuwa osobniki z prawdopodobieństwem równym (1 - fitness)
    """
    fitnesses = env.calculate_fitness()
    survives = fitnesses > torch.rand(fitnesses.shape, device=env.device)
    apply_selection(env, survives)

def apply_random_selection(env, p_survival):
    """
    Usuwa osobniki z prawdopodobieństwem równym (1 - p_survival)
    """
    survives = torch.rand(env.pop.size, device=env.device) < p_survival
    apply_selection(env, survives)

def apply_selection(env, survives):
    env.pop.genotypes = torch.masked_select(env.pop.genotypes, survives.unsqueeze(1).unsqueeze(2).expand(-1,env.pop.genotypes.shape[1],2)).reshape(
                                                -1, env.pop.genotypes.shape[1], env.pop.genotypes.shape[2]) # Usuwamy wybrane osobniki
    env.pop.size = env.pop.genotypes.shape[0]
    env.pop.positions = torch.masked_select(env.pop.positions, survives.unsqueeze(1).expand(-1,2)).reshape(-1, 2)
