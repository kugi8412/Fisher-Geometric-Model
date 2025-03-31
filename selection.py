import torch


def apply_fitness_selection(env):
    """ Removes individuals with probability equal to (1 - fitness)
    """
    fitnesses = env.calculate_fitness()
    survives = fitnesses > torch.rand(fitnesses.shape, device=env.device)
    apply_selection(env, survives)

def apply_random_selection(env,
                           p_survival: float):
    """ Removes individuals with probability equal to (1 - p_survival)
    """
    survives = torch.rand(env.pop.size, device=env.device) < p_survival
    apply_selection(env, survives)

def apply_selection(env,
                    survives: torch.tensor):
    """ Function that removes selected individuals,
    depending on the method selected by the user.
    """
    env.pop.genotypes = env.pop.genotypes[survives]
    env.pop.positions = env.pop.positions[survives]
    env.pop.size = env.pop.genotypes.shape[0]
