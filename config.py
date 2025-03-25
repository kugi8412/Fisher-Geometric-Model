import torch

# Removed device assignment here
GENOTYPE_MATRIX = torch.tensor([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]], device='cuda').T

# AREA PARAMETERS
area_width = 100.0
area_height = 100.0
size = 2
n = 4
device = 'cuda'


genotypes = torch.rand(size, n-1, 2, device=device)
last_gene_values = torch.randint(0, 2, (size, 1), device=device)
last_gene = last_gene_values.unsqueeze(2).expand(-1, -1, 2)
genotypes = torch.cat([genotypes, last_gene], dim=1)

# print(genotypes)
genes = genotypes[:, :-1].mean(-1).unsqueeze(0)
print(genotypes)
phenos = torch.matmul(genes, GENOTYPE_MATRIX)

'''

positions = torch.rand(size, 2, device=device) * torch.tensor(
            [area_width, area_height], device=device)

print(positions)'
'''
sigma = 1

optimal_genotype = torch.rand((1, n-1, 2), device=device)
print(optimal_genotype)
optimal_pheno = torch.matmul(optimal_genotype.mean(-1), GENOTYPE_MATRIX)
print('Fenotypy', phenos)
print('Fajny Geny', optimal_genotype)
print('Optimum', optimal_pheno)
phenos[:, 0] = optimal_pheno
phenos[:, 1] = -optimal_pheno
phenotype_dist = torch.norm(phenos - optimal_pheno, dim=-1)
phenotype_fitness = torch.exp(-phenotype_dist**2/(2* sigma**2))
print('Fitness', phenotype_fitness)

""" OLD MAIN
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Dict
from mutation import mutate_population
from reproduction import reproduce
from environment import FisherEnvironment
from visualization import plot_phenotype_space, plot_reproduction_space, plot_gene_history
from selection import apply_fitness_selection

def generate_phenotype_matrix(n_genes: int, device: torch.device) -> torch.Tensor:
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

def run_parameter_analysis(base_params: Dict):
    st.sidebar.subheader("Analiza parametrów")
    param1 = st.sidebar.selectbox("Parametr 1", ['mutation_rate', 'selection'])
    param2 = st.sidebar.selectbox("Parametr 2", ['opt_drift', 'mutation_strength'])
    
    grid_size = st.sidebar.slider("Rozmiar siatki", 2, 10, 5)
    trials = st.sidebar.slider("Liczba prób", 1, 10, 3)
    
    # Generowanie siatki wartości
    values1 = np.linspace(base_params[param1]*0.5, base_params[param1]*1.5, grid_size)
    values2 = np.linspace(base_params[param2]*0.5, base_params[param2]*1.5, grid_size)
    
    results = np.zeros((grid_size, grid_size))
    
    with st.expander("Mapa ciepła", expanded=True):
        placeholder = st.empty()
        progress = st.progress(0)
        total_points = grid_size**2
        
        for i, (v1, v2) in enumerate(list(values1, values2)):
            params = base_params.copy()
            params[param1] = v1
            params[param2] = v2
            
            survivals = [run_simulation(params)[-1] for _ in range(trials)]
            results[i//grid_size, i%grid_size] = np.mean(survivals)
            
            progress.progress((i+1)/total_points)
        
        fig = px.imshow(
            results,
            x=values2,
            y=values1,
            labels={'x': param2, 'y': param1},
            color_continuous_scale='Viridis',
            title=f"Przeżywalność dla {param1} vs {param2}"
        )
        placeholder.plotly_chart(fig)


def run_simulation(params):
    device = params['device']
    env = FisherEnvironment(params, device)
    gene_history = np.zeros((params['steps'], params['n_genes'] - 1))
    
    pheno_container = st.empty()
    repro_container = st.empty()
    gene_container = st.empty()

    population_history = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(params['steps']):
        phenos = env.pop.get_phenotypes()
        
        # Ruch osobników
        angles = torch.rand(env.pop.size, device=device) * 2 * torch.pi
        dx = phenos[:, 0] * torch.cos(angles)
        dy = phenos[:, 0] * torch.sin(angles)
        displacement = torch.stack([dx, dy], dim=1)
        env.pop.update_positions(displacement)

        # Mutacja
        mutate_population(env.pop, params['mutation_rate'], params['gene_mutation_rate'], params['mutation_strength'])

        # Selekcja
        apply_fitness_selection(env)
        if env.pop.size < 2:
            st.warning("Zbyt mało osobników – symulacja zakończona.")
            break

        # Reprodukcja
        reproduce(env)

        # Aktualizacja optimum
        env.update_optimal()

        # Historia genów
        gene_history[step] = env.pop.genotypes[:, :-1, :].mean(dim=(0, 2)).detach().cpu().numpy()

        # Wizualizacja
        fig_pheno = plot_phenotype_space(env)
        fig_repro = plot_reproduction_space(env.pop)
        fig_genes = plot_gene_history(gene_history[:step+1])

        pheno_container.pyplot(fig_pheno)
        repro_container.pyplot(fig_repro)
        gene_container.pyplot(fig_genes)

        population_history.append(env.population.size)
        
        # Aktualizacja progress baru
        progress = (step+1) / params['steps']
        progress_bar.progress(progress)
        status_text.text(f"Pokolenie: {step+1}/{params['steps']} | Populacja: {env.population.size}")
    
    return population_history

def main():
    st.set_page_config(page_title="Population Simulation",
                       page_icon=":cat:")

    st.title("Population Simulation")
    
    # Wybór urządzenia
    device_options = ['CPU', 'GPU'] if torch.cuda.is_available() else ['CPU']
    device = torch.device('cuda' if st.sidebar.radio("Urządzenie", device_options) == 'GPU' else 'cpu')

    # Parametry
    n_genes = st.sidebar.slider("Liczba genów", 3, 50, 11)
    with st.sidebar.expander("Zaawansowane"):
        phenotype_matrix = generate_phenotype_matrix(n_genes-1, device)

    params = {
        'device': device,
        'n_organisms': st.sidebar.slider("Liczba organizmów", 100, 10000, 2000),
        'max_population': st.sidebar.slider("Maksymalna populacja", 100, 10000, 2000),
        'n_genes': n_genes,
        'area_width': st.number_input("Area Width", value=100),
        'area_height': st.number_input("Area Height", value=100),
        'phenotype_matrix': phenotype_matrix,
        'selection': st.sidebar.slider("Siła selekcji (sigma)", 0.1, 10.0, 2.0),
        'opt_drift': st.sidebar.slider("Dryf optimum", 0.0, 2.0, 0.05),
        'opt_noise': st.sidebar.slider("Szum optimum", 0.0, 1.0, 0.1),
        'steps': st.sidebar.slider("Liczba pokoleń", 10, 1000, 500),
        'mutation_rate': st.sidebar.slider("Prawdopodobieństwo mutacji", 0.0, 1.0, 0.6),
        'gene_mutation_rate': st.sidebar.slider("Prawdopodobieństwo mutacji genu", 0.0, 1.0, 0.6),
        'mutation_strength': st.sidebar.slider("Siła mutacji", 0.0, 1.0, 0.2),
    }
    
    if st.button("Start symulacji"):
        with st.spinner("Symulacja w toku..."):
            env, gene_history = run_simulation(params)
        st.success("Symulacja zakończona!")
        
if __name__ == "__main__":
    main()

"""