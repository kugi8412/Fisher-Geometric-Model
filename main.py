import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mutation import mutate_population
from reproduction import reproduce
from environment import FisherEnvironment
from visualization import plot_phenotype_space, plot_reproduction_space, plot_gene_history
from selection import apply_fitness_selection

def generate_phenotype_matrix(n_genes: int, device: torch.device) -> torch.Tensor:
    """Dynamicznie generuje macierz transformacji genotyp->fenotyp z wagami"""
    with st.sidebar.expander("Wagi w macierzy transformacji"):
        mid = n_genes // 2
        st.markdown("**Pierwsza oś fenotypu (prędkość)**")
        weights_x = [st.slider(f"Gen {i+1}", 0.0, 2.0, 1.0, key=f"weight_x_{i}") for i in range(mid)]
        
        st.markdown("**Druga oś fenotypu (promień rozm.)**")
        weights_y = [st.slider(f"Gen {i+mid+1}", 0.0, 2.0, 1.0, key=f"weight_y_{i}") for i in range(n_genes - mid)]

    matrix = torch.zeros((n_genes, 2), device=device)
    matrix[:mid, 0] = torch.tensor(weights_x, device=device)
    matrix[mid:, 1] = torch.tensor(weights_y, device=device)
    return matrix

def run_simulation(params):
    device = params['device']
    env = FisherEnvironment(params, device)
    gene_history = np.zeros((params['steps'], params['n_genes'] - 1))
    
    pheno_container = st.empty()
    repro_container = st.empty()
    gene_container = st.empty()
    
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

    return env, gene_history

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
