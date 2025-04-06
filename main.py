import torch
import streamlit as st
from config import delete_temp_files
from analyses import run_simulation, run_parameter_analysis


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
                w2 = st.slider(f"Gene {i+1} - Radius", 0.0, 2.0, 1.0, key=f"gene_{i}_y")
            weights.append([w1, w2])
    
    return torch.tensor(weights, device=device)


def main():
    """ Main Streamlit User Interface.
    """
    st.set_page_config(page_title="Simulation", page_icon=":cat:")

    st.title("Evolving population simulation")

    # Description of parameters
    with st.expander("ℹ️ Help - Description of parameters", expanded=False):
        st.markdown("""
        **Simulation Parameters:**
        
        🧬 **Number of Genes**:  
        The number of genes influencing the phenotype (the last gene only determines sex).
                    
        🔍 **Weight of genes**:
        The multiplicative value of each gene (allele average) in how it affects
        a trait (speed or reproduction error).
        
        🧪 **Gene Drift**:  
        The direction of change in the optimal environmental genotype for each allele.

        🐇 **Initial Population**:  
        The number of organisms at the start.

        🚧 **Maximum Population**:  
        The limit of organisms in the environment. 
        In case of excess, the excess is removed.

        ⚖️ **Selection Type**:  
        - *Fitness* - the best-adapted (highest fitness) individuals survive. 
        - *Random* - survival probability is determined by the slider.

        ⏳ **Control of selection strength**:  
        Controls the strength of selection (larger means weaker selection)

        🌪️ **Optimum Noise**:  
        The degree of fluctuation in the optimal phenotype at each step.

        🎲 **Mutations**:  
        - Probability of organism mutation  
        - Probability of each allel mutation in mutated organism
        - Mutation variance (change of allel is based on Normal Distribution)

        🔍 **Reproduction Radius**:  
        A multiplier for the reproduction range (from phenotype).

        💞 **Reproduction Factors**:  
        Factors influencing reproduction chances:  
        - Fitness - general adaptation level
        - Fitness Threshold - minimum value required for reproduction
        - Capacity - dependence on maximum population (logistic model)
        
        📊 **Time Interval**:  
        How often to update visualizations in plots and GIFs.
        """)

    # Device selection
    device_options = ['CPU', 'GPU'] if torch.cuda.is_available() else ['CPU']
    device = torch.device('cuda' if st.sidebar.radio("DEVICE", device_options) == 'GPU' else 'cpu')

    # Parameters
    n_genes = st.sidebar.slider("Number of genes", 3, 11, 5)
    phenotype_matrix = generate_phenotype_matrix(n_genes-1, device)
    
    # User selection for drift for each gene affecting the phenotype
    with st.sidebar.expander("Drift for genes"):
        opt_drift = []
        for i in range(n_genes - 1):
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
        'steps': st.number_input("Number of generations", 1, 10000, 50),
        'area_width': st.number_input("Area Width", min_value=10, max_value=10000, value=100),
        'area_height': st.number_input("Area Height", min_value=10, max_value=10000, value=100),
        'n_organisms': st.sidebar.number_input("Number of organisms", 10, 100000, 1000),
        'max_population': st.sidebar.number_input("Maximum population", 10, 100000, 4000),
        'selection_type': st.sidebar.selectbox("Type of selection", ['fitness', 'random']),
        'selection': st.sidebar.slider("Controlling the strength in fitness selection", 0.1, 10.0, 1.0),
        'survival_rate': st.sidebar.slider("Probability of survival in random selection", 0.00, 1.0, 0.5),
        'opt_noise': st.sidebar.slider("Optimum noise", 0.0, 1.0, 0.1),
        'mutation_rate': st.sidebar.slider("Probability of organism mutation", 0.0, 1.0, 0.5),
        'gene_mutation_rate': st.sidebar.slider("Probability of allel mutation", 0.0, 1.0, 0.5),
        'mutation_strength': st.sidebar.slider("Mutation variance", 0.0, 1.0, 0.25),
        'radius_multiplier': st.sidebar.slider("Viewing radius multiplier", 0.1, 10.0, 1.0),
        'plot_interval': st.sidebar.slider("Time interval for plotting", 1, 100, 5),
        'reproduction_factors': st.sidebar.multiselect(
                                "Reproduction Factors:",
                                ['fitness', 'fitness_threshold', 'capacity'],
                                default=['fitness']
                                ),
        'min_fitness': st.sidebar.slider("Fitness threshold for reproduction", 0.0, 1.0, 0.5),
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
    
        grid_size1 = st.sidebar.slider("Grid size for 1 parameter", 2, 50, 20)
        grid_size2 = st.sidebar.slider("Grid size for 2 parameter", 2, 50, 20)
        trials = st.sidebar.slider("Number of trials", 1, 20, 5)
        if st.button("Start Batch Simulation"):
            run_parameter_analysis(params, param1, param2, grid_size1, grid_size2, trials)

if __name__ == "__main__":
    torch.classes.__path__ = [] # not for web streamlit
    torch.manual_seed(seed=42)

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()

    main()
    delete_temp_files("temp_frames")
