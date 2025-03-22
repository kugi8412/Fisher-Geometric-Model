import torch
import streamlit as st
from population import Population
from reproduction import sexual_reproduction
from mutation import mutate_population
from selection import selection

def main():
    st.title("Symulacja Ewolucji Płciowej")

    params = {
        'n_organisms': st.sidebar.slider("Liczba organizmów", 10, 1000, 200),
        'n_genes': st.sidebar.slider("Liczba genów", 1, 50, 10),
        'mutation_rate': st.sidebar.slider("Częstość mutacji", 0.0, 1.0, 0.1),
        'mutation_strength': st.sidebar.slider("Siła mutacji", 0.0, 1.0, 0.05),
        'speed': st.sidebar.slider("Prędkość poruszania się", 0.0, 1.0, 0.1),
        'max_population': st.sidebar.slider("Maksymalna liczba osobników", 100, 5000, 1000),
        'steps': st.sidebar.slider("Liczba pokoleń", 10, 500, 100),
        'mutations_probability': st.sidebar.slider("Prawdopodobieństwo mutacji", 0.0, 1.0, 0.10)
    }

    if st.button("Start symulacji"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pop = Population(params['n_organisms'], params['n_genes'], device)

        for step in range(params['steps']):
            pop.move_population(params['speed'])
            mutate_population(pop, params['mutation_rate'], params['mutation_strength'], params['mutations_probability'])
            selection(pop, params['alpha'], params['sigma'],  params['max_population'])
            sexual_reproduction(pop, params['alpha'], params['sigma'], params['max_population'])

        st.success("Symulacja zakończona!")

if __name__ == "__main__":
    main()
