import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List


def plot_phenotype_space(env):
    """ Matplotlib plot for phenotype space with
    radius for probability 0.5 to survive.
    """
    plt.style.use('seaborn-v0_8-dark') # Plot style
    
    pop = env.pop
    phenos = pop.get_phenotypes().detach().cpu().numpy()
    opt = env.get_optimal_phenotype().squeeze(0).detach().cpu().numpy()
    n_genes = env.params['n_genes'] - 1
    
    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Calculate the radius for a probability of 0.5
    selection_strength = env.params['selection']
    radius = selection_strength * np.sqrt(-2 * np.log(0.5))
    
    # Draw the survival zone compute early
    circle = plt.Circle((opt[0], opt[1]), radius, 
                       color='tomato', alpha=0.15, 
                       label='Fitness threshold 50%')
    ax.add_patch(circle)
    
    # Points and optimum
    ax.scatter(phenos[:, 0], phenos[:, 1], 
               c=phenos[:, 0], cmap='inferno', 
               alpha=0.5, s=100, edgecolors='w', linewidth=0.5)
    ax.scatter(opt[0], opt[1], c='red', marker='X', 
              s=400, label='Optimum', zorder=3)
    
    # Visualization of plot
    ax.set_xlim(0, n_genes * 2)
    ax.set_ylim(0, n_genes * 2)
    ax.set_xlabel('Speed', fontsize=16, labelpad=10)
    ax.set_ylabel('Reproduction range', fontsize=16, labelpad=10)
    ax.set_title(f'Phenotype space (step: {env.current_step})\n', 
                fontsize=20, pad=5)
    ax.grid(True, linestyle='--', alpha=0.75)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.9)
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_reproduction_space(env):
    """ Matplotlib plot for reproduction space.
    """
    pop = env.pop
    positions = pop.positions.detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.scatter(positions[:, 0], positions[:, 1], alpha=0.4)
    ax.set_xlim(0, pop.area_width)
    ax.set_ylim(0, pop.area_height)
    ax.set_title(f"Reproduction space (step: {env.current_step})", fontsize=16)
    return fig


def plot_gene_history_matplotlib(gene_history: np.ndarray):
    """ Matplotlib version to be updated during simulation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(gene_history.shape[1]):
        ax.plot(gene_history[:, i], label=f"Gen {i+1}")
    ax.set_xlabel("Generation", fontsize=16)
    ax.set_ylabel("Average gene value", fontsize=16)
    ax.set_title("Evolution of genes over time", fontsize=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='lower left')
    plt.tight_layout()
    return fig


def plot_population_history_plotly(population_history: List[int]):
    """ Generates an interactive graph of population size.
    """
    df = pd.DataFrame({
        'Generation': range(len(population_history)),
        'Population': population_history
    })
    
    fig = px.line(df, 
                 x='Generation', 
                 y='Population',
                 labels={'Population': 'number of individuals', 'Generation': 'step'},
                 title='Number of population over time',
                 markers=True,
                 template='plotly_white')
    
    fig.update_layout(
        hovermode='x',
        xaxis_title='Generation',
        yaxis_title='Number of individuals',
        showlegend=False
    )
    return fig


def plot_gene_history_plotly(gene_history: np.ndarray):
    """ Plotly version for final summary.
    """
    df = pd.DataFrame(gene_history)
    df.columns = [f"{i+1}" for i in range(gene_history.shape[1])]
    df['Generation'] = df.index
    
    fig = px.line(df, 
                 x='Generation', 
                 y=[col for col in df.columns if col != 'step'],
                 labels={'value': 'Average gene value', 'variable': 'Gene'},
                 title='Evolution of genes over time')
    
    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=-0.75, xanchor="auto", x=0.5),
        template='plotly_white'
    )

    '''
    fig.add_annotation(
    text="Change of mean values during time.",
    xref="paper", yref="paper",
    x=0.5, y=-0.2, 
    showarrow=False,
    font=dict(size=12)
    )
    '''
    return fig
