import os
import torch
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from itertools import product
from typing import Dict
from environment import FisherEnvironment
from reproduction import reproduce
from mutation import mutate_population
from selection import apply_fitness_selection, apply_random_selection
from visualization import plot_phenotype_space, plot_reproduction_space,\
                          plot_gene_history_plotly, plot_population_history_plotly,\
                          plot_gene_history_matplotlib, create_gif


def run_parameter_analysis(base_params: Dict[str, float],
                           param1: str, 
                           param2: str,
                           grid_size1: int,
                           grid_size2: int,
                           trials: int):
    """ Improved parameter analysis function
    to create heatmap of mean survivors.
    """
    values1 = np.linspace(0.01, 10.0, grid_size1) if param1 in [
                                                "radius_multiplier",
                                                "selection"] else np.linspace(0.01, 1.0, grid_size1)
    values2 = np.linspace(0.01, 1.0, grid_size2)
    results = np.zeros((grid_size1, grid_size2))
    total_points = grid_size1 * grid_size2
    
    with st.expander("Progress of the analysis", expanded=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (v1, v2) in enumerate(product(values1, values2)):
            params = base_params.copy()
            params[param1] = v1
            params[param2] = v2
            survivals = []

            for trial in range(trials):
                population_history = run_simulation(params, st.empty(),
                                                    st.empty(), st.empty(),
                                                    analysis=True)
                survivals.append(population_history[-1] if len(population_history) > 0 else 0)
   
            results[idx // grid_size1, idx % grid_size2] = np.mean(survivals)
            
            # Progress actualization
            progress = (idx + 1) / total_points
            progress_bar.progress(progress)
            status_text.text(f"Progress: {idx+1}/{total_points} | {param1}={v1:.2f}, {param2}={v2:.2f}")
    
    # Generating of heat map
    fig = px.imshow(
        results,
        x=values2,
        y=values1,
        labels={'x': param2, 'y': param1},
        color_continuous_scale='Inferno',
        aspect='auto'
    )

    fig.update_layout(title=f"Average Number of Survivors: {param1} vs {param2}")
    st.plotly_chart(fig)
    st.success("The analysis is complete!")


def run_simulation(params, pheno_container, repro_container, gene_container, analysis=False):
    """ Executes the main simulation loop.
    """
    # Dir for GIF-s
    os.makedirs("temp_frames", exist_ok=True)
    pheno_frames = []
    repro_frames = []

    device = params['device']
    env = FisherEnvironment(params, device)
    gene_history = np.zeros((params['steps'], params['n_genes'] - 1))
    pheno_container = st.empty()
    repro_container = st.empty()
    gene_container = st.empty()
    population_history = [env.pop.size]

    # Checking User
    if not params['reproduction_factors']:
        st.error("Choose at least one reproduction factor!")
        return

    for step in range(params['steps']):
        phenos = env.pop.get_phenotypes()
        gene_history[step] = env.pop.genotypes[:, :-1, :].mean(dim=(0, 2)).detach().cpu().numpy()
        
        # Move individuals based on phenotype speed
        angles = torch.rand(env.pop.size, device=device) * 2 * torch.pi
        dx = phenos[:, 0] * torch.cos(angles)
        dy = phenos[:, 0] * torch.sin(angles)
        displacement = torch.stack([dx, dy], dim=1)
        env.pop.update_positions(displacement)

        # Mutation
        mutate_population(env.pop, params['mutation_rate'],
                          params['gene_mutation_rate'],
                          params['mutation_strength'])

        # Selection
        if params['selection_type'] == "fitness":
            apply_fitness_selection(env)
        elif params['selection_type'] == "random":
            apply_random_selection(env, params['survival_rate'])
        

        if env.pop.size < 2:
            if not analysis:
                st.warning("Too few individuals - simulation terminated!")
            else:
                population_history.append(0) # Death of population
            break

        # Reproduction
        stop = reproduce(env)
        if stop:
            if not analysis:
                st.warning("No individuals of one of the sexes - simulation terminated!")
            else:
                population_history.append(0) # Death of population
            break

        # Update optimum
        env.update_optimal()

        # History of number of population
        population_history.append(env.pop.size)

        # Visualization (plots in plt and pyplot)
        if step % params['plot_interval'] == 0 and not analysis:

            # Generate matplotlib figures
            fig_pheno = plot_phenotype_space(env)
            fig_repro = plot_reproduction_space(env)
            fig_genes = plot_gene_history_matplotlib(gene_history[:step+1])
            
            # Update containers with matplotlib plots
            pheno_container.pyplot(fig_pheno)
            repro_container.pyplot(fig_repro)
            gene_container.pyplot(fig_genes)
            plt.close('all')

            pheno_frame_path = f"temp_frames/pheno_{step:04d}.png"
            repro_frame_path = f"temp_frames/repro_{step:04d}.png"
            fig_pheno.savefig(pheno_frame_path, bbox_inches='tight', pad_inches=0.5, dpi=100)
            fig_repro.savefig(repro_frame_path, bbox_inches='tight', pad_inches=0.5, dpi=100)
            pheno_frames.append(pheno_frame_path)
            repro_frames.append(repro_frame_path)

    # After finishing the simulation, successfully or not
    if not analysis:
        with st.expander("Interactive Plots", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_gene_history_plotly(gene_history[:step+1]))
            with col2:
                st.plotly_chart(plot_population_history_plotly(population_history))
        
        with st.expander("Animations", expanded=True):
            if len(pheno_frames) > 1:
                create_gif(pheno_frames, "phenotype_evolution.gif")
                st.image("phenotype_evolution.gif", caption="Evolution of phenotypes")
                
            if len(repro_frames) > 1:
                create_gif(repro_frames, "reproduction_space.gif")
                st.image("reproduction_space.gif", caption="Reproduction space")
            
    return population_history
