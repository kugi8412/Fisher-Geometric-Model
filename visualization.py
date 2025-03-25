import matplotlib.pyplot as plt
import imageio
import os
import torch
import matplotlib.pyplot as plt

def plot_gene_history(gene_history):
    """
    Plots the average values of genes over time.
    :param gene_history: numpy array (steps, n_genes) - history of gene averages
    """
    fig, ax = plt.subplots()
    for i in range(gene_history.shape[1]):
        ax.plot(gene_history[:, i], label=f"Gene {i+1}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Gene Value")
    ax.set_title("Gene Evolution Over Time")
    ax.legend()
    return fig


def save_frames_as_gif(frames, filename="animation.gif", duration=0.2):
    """ Saves frames as a GIF animation. """
    with imageio.get_writer(filename, mode='I', duration=duration) as writer:
        for frame in frames:
            writer.append_data(frame)

def plot_phenotype_space(env):
    phenos = env.population.get_phenotypes().detach().cpu().numpy()
    opt = env.get_optimal_phenotype().detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.scatter(phenos[:, 0], phenos[:, 1], alpha=0.5)
    ax.scatter(opt[0], opt[1], color='red', marker='X', s=150)
    
    # Dynamiczny zakres osi na podstawie liczby genów
    max_pheno = (env.params['n_genes'] - 1) // 2
    ax.set_xlim(0, max_pheno)
    ax.set_ylim(0, max_pheno)
    
    return fig

def plot_reproduction_space(env, save_frames=False):
    """ Plots reproduction space and optionally saves frames for a GIF. """
    positions = env.population.positions.detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.scatter(positions[:, 0], positions[:, 1], alpha=0.7, label="Positions")
    ax.set_xlim(0, env.population.area_width)
    ax.set_ylim(0, env.population.area_height)
    ax.set_title("Physical Reproduction Space")
    ax.legend()

    if save_frames:
        plt.savefig("frames/frame_reproduction.png", dpi=100)

    return fig
