import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

def plot_phenotype_space(env):
    """
    Rysuje przestrzeń fenotypową: każdy punkt reprezentuje wyliczony fenotyp osobnika,
    a czerwony krzyżyk z poświatą oznacza aktualne optimum (przyjmujemy, że optimum jest przechowywane w pop).
    """
    pop = env.pop
    phenos = pop.get_phenotypes().detach().cpu().numpy()
    # Zakładamy, że optimum jest atrybutem obiektu population (możesz go przekazać inaczej)
    opt = env.get_optimal_phenotype().squeeze(0) if hasattr(env, "optimal_genotype") else np.array([0,0])
    fig, ax = plt.subplots()
    ax.scatter(phenos[:, 0], phenos[:, 1], alpha=0.7, label="Organisms (phenotype)")
    # Rysujemy optimum z czerwoną obwódką
    ax.scatter(opt[0], opt[1], color='red', marker='X', s=150, edgecolor='red', linewidth=2, label="Optimum")
    ax.set_xlim(0, pop.area_width/ (pop.n_genes -1)*((pop.n_genes-1)/2))
    ax.set_ylim(0, pop.area_height/ (pop.n_genes -1)*((pop.n_genes-1)/2))
    ax.set_title("Phenotype Space")
    ax.legend()
    return fig

def plot_reproduction_space(env):
    """
    Rysuje przestrzeń fizyczną: pozycje osobników.
    """
    pop = env.pop
    positions = pop.positions.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.scatter(positions[:, 0], positions[:, 1], alpha=0.7, label="Positions")
    ax.set_xlim(0, pop.area_width)
    ax.set_ylim(0, pop.area_height)
    ax.set_title("Physical Reproduction Space")
    ax.legend()
    return fig

def plot_gene_history(gene_history):
    fig, ax = plt.subplots()
    for i in range(gene_history.shape[1]):
        ax.plot(gene_history[:, i], label=f"Gene {i+1}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Genes")
    ax.legend()
    return fig
