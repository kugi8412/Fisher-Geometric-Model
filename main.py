import os
import time
import torch
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict
from itertools import product
from environment import FisherEnvironment
from reproduction import reproduce
from mutation import mutate_population
from selection import apply_fitness_selection, apply_random_selection
from visualization import plot_phenotype_space, plot_reproduction_space,\
                          plot_gene_history_plotly, plot_population_history_plotly,\
                          plot_gene_history_matplotlib


def generate_phenotype_matrix(n_genes: int, device: torch.device) -> torch.Tensor:
    """Generates transformation
    matrix for genotype -> phenotype.
    """
    with st.sidebar.expander("Weights for Genes"):
        weights = []
        for i in range(n_genes):
            col1, col2 = st.columns(2)
            with col1:
                w1 = st.slider(f"Gene {i+1} - Speed", 0.0, 2.0, 1.0, key=f"gene_{i}_x")
            with col2:
                w2 = st.slider(f"Gene {i+1} - Reproduction", 0.0, 2.0, 1.0, key=f"gene_{i}_y")
            weights.append([w1, w2])
    
    return torch.tensor(weights, device=device)#.T.transpose(0, 1)

def run_parameter_analysis(base_params: Dict[str, float],
                           param1: str, 
                           param2: str,
                           grid_size: int,
                           trials: int):
    """ Improved parameter analysis function
    to create heatmap of mean survivors.
    """
    if param1 == "radius_multiplier" or param1 == "selection":
        values1 = np.linspace(0.0, 10.0, grid_size)
    else:
        values1 = np.linspace(0.0, 1.0, grid_size)
    
    values2 = np.linspace(0.01, 1.0, grid_size)

    results = np.zeros((grid_size, grid_size))
    total_points = grid_size ** 2
    
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
   
            results[idx // grid_size, idx % grid_size] = np.mean(survivals)
            
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
        st.error("Musisz wybrać przynajmniej jeden czynnik reprodukcji!")
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
        
    # Removing unnecessary files
    if not analysis:
        plt.close('all')
        time.sleep(0.1)
        delete_temp_files("temp_frames")
    
    return population_history


def create_gif(frame_paths,
               output_filename: str,
               duration: int = 400):
    """ Creates a GIF with a forced size consistency.
    """
    
    images = []
    base_size = (800, 800)
    
    for path in frame_paths:
        try:
            img = Image.open(path)
            img = img.resize(base_size).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error with frame {path}: {e}")
    
    if images:
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True
        )

def delete_temp_files(temp_dir: str):
    """ Safe deletion of temporary files.
    """
    max_retries = 2
    for i in range(max_retries):
        try:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(temp_dir)
            break
        except PermissionError:
            time.sleep(0.2 * (i+1)) # waiting to close all files
        except FileNotFoundError:
            break

def main():
    """ Main Streamlit User Interface.
    """
    st.set_page_config(page_title="Simulation", page_icon=":cat:")

    st.title("Evolving population simulation")
    
    # Device selection
    device_options = ['CPU', 'GPU'] if torch.cuda.is_available() else ['CPU']
    device = torch.device('cuda' if st.sidebar.radio("DEVICE", device_options) == 'GPU' else 'cpu')

    # Parameters
    n_genes = st.sidebar.slider("Number of genes", 3, 11, 4)
    phenotype_matrix = generate_phenotype_matrix(n_genes-1, device)
    
    # User selection for drift for each gene
    with st.sidebar.expander("Drift for genes"):
        opt_drift = []
        for i in range(n_genes - 1):  # For each gene affecting the phenotype
            col1, col2 = st.columns(2)
            with col1:
                drift_allele1 = st.slider(f"Gene {i+1} drift (allel 1)", -1.0, 1.0, 0.0, key=f"drift_{i}_a1")
            with col2:
                drift_allele2 = st.slider(f"Gene {i+1} drift (allel 2)", -1.0, 1.0, 0.0, key=f"drift_{i}_a2")
            opt_drift.extend([drift_allele1, drift_allele2])

    params = {
        'device': device,
        'n_genes': n_genes,
        'opt_drift': opt_drift,
        'phenotype_matrix': phenotype_matrix,
        'area_width': st.number_input("Area Width", min_value=10, max_value=10000, value=100),
        'area_height': st.number_input("Area Height", min_value=10, max_value=10000, value=100),
        'steps': st.sidebar.number_input("Number of generations", 10, 10000, 50),
        'n_organisms': st.sidebar.number_input("Number of organisms", 10, 100000, 1000),
        'max_population': st.sidebar.number_input("Maximum population", 10, 100000, 2000),
        'selection_type': st.sidebar.selectbox("Type of selection", ['fitness', 'random']),
        'selection': st.sidebar.slider("The power of selection (fitness selection)", 0.1, 10.0, 1.0),
        'survival_rate': st.sidebar.slider("Probability of survival (random selection)", 0.00, 1.0, 0.5),
        'opt_noise': st.sidebar.slider("Optimum noise", 0.0, 1.0, 0.1),
        'mutation_rate': st.sidebar.slider("Probability of organism mutation", 0.0, 1.0, 0.6),
        'gene_mutation_rate': st.sidebar.slider("Probability of gene mutation", 0.0, 1.0, 0.6),
        'mutation_strength': st.sidebar.slider("Power of mutation", 0.0, 1.0, 0.2),
        'radius_multiplier': st.sidebar.slider("Viewing radius multiplier", 0.1, 10.0, 1.0),
        'plot_interval': st.sidebar.slider("Time interval for plotting", 1, 100, 10),
        'reproduction_factors': st.sidebar.multiselect(
                                "Czynniki reprodukcji",
                                ['fitness', 'fitness_threshold', 'capacity'],
                                default=['fitness']
                                ),
        'min_fitness': st.sidebar.slider("Próg fitness do reprodukcji", 0.0, 1.0, 0.5),
    }
    
    if st.button("Start of simulation"):
        with st.spinner("Simulation in progress..."):
            pheno_container = st.empty()
            repro_container = st.empty()
            gene_container = st.empty()
            run_simulation(params, pheno_container, repro_container, gene_container)
        st.success("Simulation completed!")

    if st.sidebar.checkbox("Heat map analysis"):
        st.sidebar.subheader("Analysis of parameter change")
        param1 = st.sidebar.selectbox("Parameter 1", ['selection', 'survival_rate',
                                                      'radius_multiplier'])

        param2 = st.sidebar.selectbox("Parameter 2", ['mutation_rate', 'gene_mutation_rate', 
                                                      'mutation_strength', 'opt_noise'])
    
        grid_size = st.sidebar.slider("Grid size", 2, 20, 10)
        trials = st.sidebar.slider("Number of trials", 1, 20, 5)
        if st.button("Start Batch Simulation"):
            run_parameter_analysis(params, param1, param2, grid_size, trials)

if __name__ == "__main__":
    main()
