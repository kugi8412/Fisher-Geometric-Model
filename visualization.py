import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from typing import List


def plot_phenotype_space(env):
    """ Matplotlib plot for phenotype space with
    radius for probability 0.5 to survive.
    """
    plt.style.use('seaborn-v0_8-dark')
    
    pop = env.pop
    phenos = pop.get_phenotypes().detach().cpu().numpy()
    opt = env.get_optimal_phenotype().squeeze(0).detach().cpu().numpy()
    limes = env.params['phenotype_matrix'].sum(axis=0).squeeze()
    max_speed, max_radius = int(limes[0]), int(limes[1])
    
    fig, ax = plt.subplots(figsize=(9, 7), dpi=400)
    plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
    
    # Calculate distances and fitness
    distances = np.linalg.norm(phenos - opt, axis=1)
    fitness = np.exp(-distances**2 / (2 * env.params['selection']))

    '''
    selection_strength = env.params['selection']
    radius = selection_strength * np.sqrt(-2 * np.log(0.5))
    circle = plt.Circle((opt[0], opt[1]), radius, 
                       color='tomato', alpha=0.15, 
                       label='50% survival threshold')
    ax.add_patch(circle)
    '''
    
    # Fitness colour scatter plot
    sc = ax.scatter(
        phenos[:, 0], phenos[:, 1], 
        c=fitness,
        cmap='RdYlGn',
        alpha=0.6,
        s=100,
        vmin=0,
        vmax=1
    )
    
    # Optimal point
    ax.scatter(opt[0], opt[1], marker='X', 
              s=600, color='#0000FF', label='Optimum', zorder=1)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Fitness', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    # Labels and styling
    ax.set_xlim(0, max_speed + 0.5)
    ax.set_ylim(0, max_radius + 0.5)
    ax.set_xlabel('Speed', fontsize=16)
    ax.set_ylabel('Reproduction Range', fontsize=16, labelpad=10)
    ax.set_title(f'Phenotype Space (generation: {env.current_step})', fontsize=20, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Legend
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        frameon=True,
        framealpha=0.9,
        fontsize=12
    )
    
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
    ax.set_title(f"Reproduction space (generation: {env.current_step})", fontsize=16)
    return fig


def plot_gene_history_matplotlib(gene_history: np.ndarray):
    """ Matplotlib version to be updated during simulation.
    """
    fig, ax = plt.subplots()
    for i in range(gene_history.shape[1]):
        ax.plot(gene_history[:, i], label=f"Gen {i+1}")

    ax.set_xlabel("Generation", fontsize=14)
    ax.set_ylabel("Average gene value", fontsize=14)
    ax.set_title("Evolution of genes over time", fontsize=18)
    ax.legend(bbox_to_anchor=(1.00, 1.00), loc='lower left')
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
                 labels={'Population': 'Number of individuals', 'Generation': 'Generation'},
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


def create_gif(frame_paths: list,
               output_filename: str,
               duration: int = 400):
    """Creates GIF from stored image files.
    
    Args:
        frame_paths (list): List of paths to individual frames
        output_filename (str): Output GIF filename
        duration (int): Display duration per frame in milliseconds
    """
    
    images = []
    base_size = (800, 800)
    
    for path in frame_paths:
        try:
            # Open frames
            with Image.open(path) as img:
                img = img.resize(base_size).convert('RGB')
                images.append(img.copy())
        except Exception as e:
            print(f"Error processing frames {path}: {e}")
            continue
    
    if images:
        try:
            # Make GIF, from first image
            images[0].save(
                output_filename,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
        except Exception as e:
            print(f"Failed to save GIF: {e}")
