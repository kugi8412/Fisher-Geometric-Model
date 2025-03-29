import torch
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict
from mutation import mutate_population
from reproduction import reproduce
from environment import FisherEnvironment
from visualization import plot_phenotype_space, plot_reproduction_space, plot_gene_history
from selection import apply_fitness_selection, apply_random_selection
from itertools import product
import matplotlib.pyplot as plt

def generate_phenotype_matrix(n_genes: int, device: torch.device) -> torch.Tensor:
    """Generates transformation matrix for genotype -> phenotype"""
    with st.sidebar.expander("Wagi genów (2 osie)"):
        weights = []
        for i in range(n_genes):
            col1, col2 = st.columns(2)
            with col1:
                w1 = st.slider(f"Gen {i+1} - Oś X", 0.0, 2.0, 1.0, key=f"gene_{i}_x")
            with col2:
                w2 = st.slider(f"Gen {i+1} - Oś Y", 0.0, 2.0, 1.0, key=f"gene_{i}_y")
            weights.append([w1, w2])
    
    return torch.tensor(weights, device=device).T

def run_parameter_analysis(base_params: Dict, param1: str, param2: str, grid_size: int, trials: int):
    """Runs multiple simulations for parameter analysis and generates heatmap."""
    
    values1 = np.linspace(base_params[param1]*0.5, base_params[param1]*1.5, grid_size)
    values2 = np.linspace(base_params[param2]*0.5, base_params[param2]*1.5, grid_size)
    
    results = np.zeros((grid_size, grid_size))
    
    with st.expander("Mapa ciepła", expanded=True):
        placeholder = st.empty()
        progress = st.progress(0)
        total_points = grid_size**2
        progress.progress((i+1)/total_points, f"{param1}: {v1}, {param2}: {v2} iteration {i+1}/{total_points}")
        
        for i, (v1, v2) in enumerate(product(values1, values2)):
            params = base_params.copy()
            params[param1] = v1
            params[param2] = v2
            
            survivals = []  # Take last population size
            progress_trial = st.progress(0)
            for k in range(trials):
                progress_trial.progress((k+1)/trials,f"trial {k+1}/{trials}")
                pheno_container = st.empty()
                repro_container = st.empty()
                gene_container = st.empty()
                pop_container = st.empty()
                survivals.append(run_simulation(params, pheno_container, repro_container, gene_container, pop_container, analysis=True)[-1])
                pheno_container.empty()
                repro_container.empty()
                gene_container.empty()
                pop_container.empty()
                plt.close("all")
            progress_trial.empty()

            results[i//grid_size, i%grid_size] = np.mean(survivals)

        fig = px.imshow(
            results,
            x=values2,
            y=values1,
            labels={'x': param2, 'y': param1, 'color': 'Population'},
            color_continuous_scale='Viridis',
            title=f"Przeżywalność dla {param1} vs {param2}",
            aspect='auto',
            width=1600,
            height=600
        )
        fig.update_layout(xaxis_title=param2, yaxis_title=param1)
        placeholder.plotly_chart(fig)

def run_simulation(params, pheno_container, repro_container, gene_container, pop_container, analysis=False):
    """Executes the main simulation loop."""
    device = params['device']
    env = FisherEnvironment(params, device)
    gene_history = np.zeros((params['steps'], params['n_genes'] - 1))

    # pheno_container = st.empty()
    # repro_container = st.empty()
    # gene_container = st.empty()
    # pop_container = st.empty()

    population_history = []
    if not analysis:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for step in range(params['steps']):
        phenos = env.pop.get_phenotypes()
        
        # Move individuals based on phenotype speed
        angles = torch.rand(env.pop.size, device=device) * 2 * torch.pi
        dx = phenos[:, 0] * torch.cos(angles)
        dy = phenos[:, 0] * torch.sin(angles)
        displacement = torch.stack([dx, dy], dim=1)
        env.pop.update_positions(displacement)

        # Mutation
        mutate_population(env.pop, params['mutation_rate'], params['gene_mutation_rate'], params['mutation_strength'])

        # Selection
        if params['selection_type'] == 'fitnessowa':
            apply_fitness_selection(env)
        elif params['selection_type'] == 'losowa':
            apply_random_selection(env, params['survival_rate'])
        
        if env.pop.size < 2:
            if not analysis:
                st.warning("Zbyt mało osobników – symulacja zakończona.")
            break

        # Reproduction
        stop = reproduce(env)
        if stop:
            if not analysis:
                st.warning("Brak osobników jednej z płci - symulacja zakończona.")
            break

        # Update optimum
        env.update_optimal()

        # Log history
        gene_history[step] = env.pop.genotypes[:, :-1, :].mean(dim=(0, 2)).detach().cpu().numpy()
        population_history.append(env.pop.size)

        # Visualization TO CHANGE
        if step % params['plot_interval'] == 0 and not analysis:
            fig_pheno = plot_phenotype_space(env)  # macierz transformacji już uwzględniona w get_phenotypes
            fig_repro = plot_reproduction_space(env)
            fig_genes = plot_gene_history(gene_history[:step])
            pheno_container.pyplot(fig_pheno)
            repro_container.pyplot(fig_repro)
            gene_container.pyplot(fig_genes, use_container_width=True)
    
    return population_history

def main():
    """Main Streamlit UI."""
    st.set_page_config(page_title="Population Simulation", page_icon=":cat:")

    st.title("Population Simulation")
    
    # Device selection
    device_options = ['CPU', 'GPU'] if torch.cuda.is_available() else ['CPU']
    device = torch.device('cuda' if st.sidebar.radio("Urządzenie", device_options) == 'GPU' else 'cpu')

    # Parameters
    n_genes = st.sidebar.slider("Liczba genów", 3, 50, 11)
    phenotype_matrix = generate_phenotype_matrix(n_genes-1, device)

    params = {
        'device': device,
        'n_organisms': st.sidebar.slider("Liczba organizmów", 10, 10000, 2000),
        'max_population': st.sidebar.slider("Maksymalna populacja", 10, 10000, 2000),
        'n_genes': n_genes,
        'area_width': st.number_input("Area Width", value=100),
        'area_height': st.number_input("Area Height", value=100),
        'phenotype_matrix': phenotype_matrix,
        'selection': st.sidebar.slider("Siła selekcji (selekcja fitnessowa)", 0.1, 10.0, 2.0),
        'survival_rate': st.sidebar.slider("Prawdopodobieństwo przezycia (selekcja losowa)", 0.1, 1.0, 0.9),
        'opt_drift': st.sidebar.slider("Dryf optimum", 0.0, 2.0, 0.05),
        'opt_noise': st.sidebar.slider("Szum optimum", 0.0, 1.0, 0.1),
        'steps': st.sidebar.slider("Liczba pokoleń", 10, 10000, 500),
        'mutation_rate': st.sidebar.slider("Prawdopodobieństwo mutacji", 0.0, 1.0, 0.6),
        'gene_mutation_rate': st.sidebar.slider("Prawdopodobieństwo mutacji genu", 0.0, 1.0, 0.6),
        'mutation_strength': st.sidebar.slider("Siła mutacji", 0.0, 1.0, 0.2),
        'plot_interval': st.sidebar.slider("Interwał wykresów", 1, 100, 10),
        'selection_type': st.sidebar.selectbox("Typ selekcji", ['fitnessowa', 'losowa']),
        'radius_multiplier': st.sidebar.slider("Mnożnik promienia", 0.0, 10.0, 1.0),
    }
    
    if st.button("Start symulacji"):
        with st.spinner("Symulacja w toku..."):
            pheno_container = st.empty()
            repro_container = st.empty()
            gene_container = st.empty()
            pop_container = st.empty()
            run_simulation(params, pheno_container, repro_container, gene_container, pop_container)
        st.success("Symulacja zakończona!")

    if st.sidebar.checkbox("Tryb batch (Mapa ciepła)"):
        st.sidebar.subheader("Analiza parametrów")
        param1 = st.sidebar.selectbox("Parametr 1", ['mutation_rate', 'selection'])
        param2 = st.sidebar.selectbox("Parametr 2", ['opt_drift', 'mutation_strength'])
    
        grid_size = st.sidebar.slider("Rozmiar siatki", 2, 10, 5)
        trials = st.sidebar.slider("Liczba prób", 1, 10, 3)
        if st.button("Start Batch Simulation"):
            run_parameter_analysis(params, param1, param2, grid_size, trials)

if __name__ == "__main__":
    main()
