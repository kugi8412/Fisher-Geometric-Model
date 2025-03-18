import os
import time
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mutation import mutate_population
from reproduction import reproduce
from environment import FisherEnvironment
from visualization import plot_phenotype_space, plot_reproduction_space, plot_gene_history
from selection import apply_fitness_selection
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_simulation(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FisherEnvironment(params, device)
    gene_history = {f'Gen {i+1}': [] for i in range(params['n_genes'] - 1)}
    
    pheno_container = st.empty()
    repro_container = st.empty()
    gene_container = st.empty()
    
    for step in range(params['steps']):
        # Ruch: przeliczamy fenotypy, a następnie ruch zależy od pierwszej składowej fenotypu
        phenos = env.pop.get_phenotypes()
        angles = torch.rand(env.pop.size, device=device) * 2 * np.pi
        dx = phenos[:, 0] * torch.cos(angles)
        dy = phenos[:, 0] * torch.sin(angles)
        displacement = torch.stack([dx, dy], dim=1)
        env.pop.update_positions(displacement)
        
        # Mutacja
        mutate_population(env.pop, params['mutation_rate'], params['mutation_strength'])
        
        # Selekcja – fitness-owa
        survivors = apply_fitness_selection(env.pop, env.optimal.detach().cpu().numpy(), params['selection'])
        if len(survivors) < 2:
            st.warning("Zbyt mało osobników – symulacja zakończona.")
            break
        env.pop.individuals = survivors
        env.pop.size = len(survivors)
        
        # Reprodukcja
        reproduce(env)
        # Aktualizacja optimum
        env.update_optimal()
        
        # Logowanie historii genów
        for j in range(params['n_genes'] - 1):
            gene_vals = [ind.genotype[j].mean().item() for ind in env.pop.individuals]
            avg_val = sum(gene_vals) / len(gene_vals)
            gene_history[f'Gen {j+1}'].append(avg_val)
        
        fig_pheno = plot_phenotype_space(env.pop)  # macierz transformacji już uwzględniona w get_phenotypes
        fig_repro = plot_reproduction_space(env.pop)
        fig_genes = plot_gene_history(gene_history)
        pheno_container.pyplot(fig_pheno)
        repro_container.pyplot(fig_repro)
        gene_container.plotly_chart(fig_genes, use_container_width=True)
        
        time.sleep(0.1)
    
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
    
    params = {
        'n_organisms': st.sidebar.slider("Liczba organizmów", 100, 10000, 2000),
        'n_genes': 11,
        'area_width': st.sidebar.slider("Szerokość obszaru (pozycje)", 10, 100, 100),
        'area_height': st.sidebar.slider("Wysokość obszaru (pozycje)", 10, 100, 100),
        'selection': st.sidebar.slider("Siła selekcji (sigma)", 0.1, 5.0, 1.0),
        'opt_drift': st.sidebar.slider("Dryf optimum", 0.0, 2.0, 0.3),
        'opt_noise': st.sidebar.slider("Szum optimum", 0.0, 1.0, 0.1),
        'steps': st.sidebar.slider("Liczba pokoleń", 10, 1000, 500),
        'mutation_rate': st.sidebar.slider("Prawdopodobieństwo mutacji", 0.0, 1.0, 0.1),
        'mutation_strength': st.sidebar.slider("Siła mutacji", 0.0, 1.0, 0.05)
    }
    
    if st.button("Start symulacji"):
        with st.spinner("Symulacja w toku..."):
            env, gene_history = run_simulation(params)
        st.success("Symulacja zakończona!")
        
if __name__ == "__main__":
    main()
