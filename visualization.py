import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

def plot_phenotype_space(env):
    pop = env.pop
    phenos = pop.get_phenotypes().detach().cpu().numpy()
    opt = env.get_optimal_phenotype().detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.scatter(phenos[:, 0], phenos[:, 1], alpha=0.5, label="Organizmy")
    ax.scatter(opt[0][0], opt[0][1], color='red', marker='X', s=100, label="Optimum")
    
    # Dynamiczne zakresy
    x_min, x_max = phenos[:,0].min(), phenos[:,0].max()
    y_min, y_max = phenos[:,1].min(), phenos[:,1].max()
    x_pad = max(1.0, (x_max - x_min) * 0.2)
    y_pad = max(1.0, (y_max - y_min) * 0.2)
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.legend()
    return fig

def plot_reproduction_space(pop):
    """
    Rysuje przestrzeń fizyczną: pozycje osobników.
    """
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
