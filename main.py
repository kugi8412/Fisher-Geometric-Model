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
    # Create expander for matrix weights
    with st.sidebar.expander("Wagi w macierzy transformacji"):
        # Dzielenie genów na dwie grupy
        mid = n_genes // 2
        st.markdown("**Pierwsza oś fenotypu (prędkość)**")
        weights_x = [st.slider(f"Gen {i+1}", 0.0, 2.0, 1.0, key=f"weight_x_{i}") for i in range(mid)]
        
        st.markdown("**Druga oś fenotypu (promień rozm.)**")
        weights_y = [st.slider(f"Gen {i+mid+1}", 0.0, 2.0, 1.0, key=f"weight_y_{i}") for i in range(n_genes - mid - 1)]

    # Budowa macierzy z wagami
    matrix = torch.zeros((n_genes - 1, 2), device=device)
    matrix[:mid, 0] = torch.tensor(weights_x, device=device)  # Wagi dla pierwszej osi
    matrix[mid:, 1] = torch.tensor(weights_y, device=device)  # Wagi dla drugiej osi
    
    return matrix

def run_simulation(params):
    device = params['device']
    env = FisherEnvironment(params, device)
    
    gene_history = np.zeros((params['steps'], params['n_genes'] - 1))  # Track gene changes over time
    
    pheno_container = st.empty()
    repro_container = st.empty()
    gene_container = st.empty()
    
    for step in range(params['steps']):
        # Compute phenotypes based on genotypes
        phenos = env.population.get_phenotypes()
        
        # Movement: individuals move based on their speed (phenotype component 0)
        angles = torch.rand(env.population.size, device=device) * 2 * torch.pi
        dx = phenos[:, 0] * torch.cos(angles)
        dy = phenos[:, 0] * torch.sin(angles)
        displacement = torch.stack([dx, dy], dim=1)
        env.population.update_positions(displacement)

        # Mutation
        mutate_population(env.population, params['mutation_rate'], params['gene_mutation_rate'], params['mutation_strength'])

        # Selection – fitness-based
        apply_fitness_selection(env)
        if env.population.size < 2:
            st.warning("Zbyt mało osobników – symulacja zakończona.")
            break

        # Reproduction
        reproduce(env)

        # Update environment optimum
        env.update_optimal()

        # Log gene history
        gene_history[step] = env.population.genotypes[:, :-1, :].mean(dim=(0, 2)).detach().cpu().numpy()

        # Generate and display plots
        fig_pheno = plot_phenotype_space(env)
        fig_repro = plot_reproduction_space(env)
        fig_genes = plot_gene_history(gene_history[:step])

        pheno_container.pyplot(fig_pheno)
        repro_container.pyplot(fig_repro)
        gene_container.pyplot(fig_genes, use_container_width=True)

    return env, gene_history

def main():
    st.title("Symulacja Ewolucji Fisherowskiej (Live Updates)")
    st.markdown("""
    Symulacja z live-updated plots.
    
    Każdy osobnik ma 11 genów (0..9 – fenotypowe, 10 – płciowy).  
    Fenotyp (prędkość i promień rozmnażania) obliczany jest na podstawie sumy genów – 
    dzielimy 10 genów na dwie grupy (po 5 na oś). Maksymalna wartość na osi wynosi 5.
    Optimum porusza się zgodnie z rozkładem normalnym (parametry dryfu i szumu podawane są przez użytkownika).
    """)

    # Device selection
    if torch.cuda.is_available():
        device_options = ['CPU', 'GPU']
        default_device = 'GPU'
    else:
        device_options = ['CPU']
        default_device = 'CPU'
    
    selected_device = st.sidebar.radio("Wybierz urządzenie", device_options, index=device_options.index(default_device))
    device = torch.device('cuda' if selected_device == 'GPU' else 'cpu')

    n_genes = st.sidebar.slider("Liczba genów", 3, 50, 11)
    area_width = st.sidebar.slider("Szerokość obszaru (pozycje)", 10, 100, 50)
    area_height = st.sidebar.slider("Wysokość obszaru (pozycje)", 10, 100, 50)

    with st.sidebar:
        st.header("Konfiguracja macierzy transformacji")
        phenotype_matrix = generate_phenotype_matrix(n_genes - 1, device)

    params = {
        'device': device,
        'n_organisms': st.sidebar.slider("Liczba organizmów", 100, 10000, 2000),
        'n_genes': n_genes,
        'area_width': area_width,
        'area_height': area_height,
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
