# main.py

import torch
from environment import Environment
from population import Population
from mutation import mutate_population
from selection import selection
from reproduction import asexual_reproduction
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    st.title("Symulacja Ewolucji Według Modelu Fishera")
    st.markdown("""
    Symulacja populacji organizmów ewoluujących w kierunku zmieniającego się optimum fenotypowego.
    Każdy gen składa się z dwóch alleli (A/a), których wartości wpływają na fenotyp.
    """)

    params = {
        'n_organisms': st.sidebar.slider("Liczba organizmów", 10, 1000, 200),
        'n_genes': st.sidebar.slider("Liczba genów", 1, 50, 10),
        'selection': st.sidebar.slider("Siła selekcji", 0.1, 10.0, 1.0),
        # 'selection': st.sidebar.select_slider("Siła selekcji", [10**i for i in range(-10, 5)], 1.0),
        'threshold': st.sidebar.slider("Próg sekekcji", 0.0, 1.0, 0.1),
        'mutation_rate': st.sidebar.slider("Częstość mutacji", 0.0, 1.0, 0.1),
        'gene_mutation_rate': st.sidebar.slider("Częstość mutacji genów", 0.0, 1.0, 0.1),
        'mutation_mag': st.sidebar.slider("Siła mutacji", 0.0, 1.0, 0.1),
        # 'reproduction_rate': st.sidebar.slider("Współczynnik reprodukcji", 0.0, 1.0, 0.5),
        'opt_drift': st.sidebar.slider("Dryf optimum", -1.0, 1.0, 0.1),
        'opt_noise': st.sidebar.slider("Zmienność optimum", 0.0, 1.0, 0.1),
        'steps': st.sidebar.slider("Liczba pokoleń", 10, 1000, 100),
        'log_interval': st.sidebar.slider("Interwał logowania", 1, 50, 5)
    }

    if st.button("Rozpocznij symulację"):
        with st.spinner("Przebieg symulacji..."):
            html = run_simulation(params)
        st.success("Symulacja zakończona!")
        components.html(html, width=800, height=600)

    st.markdown("""
    ### Opis parametrów:
    - **Liczba organizmów**: Początkowa wielkość populacji
    - **Liczba genów**: Wymiarowość przestrzeni fenotypowej
    - **Siła selekcji**: Im większa wartość, tym silniejsza presja selekcyjna
    - **Częstość/Siła mutacji**: Prawdopodobieństwo i wielkość zmian w allelach
    - **Dryf optimum**: Średnia zmiana optimum fenotypowego w każdym kroku
    - **Zmienność optimum**: Losowa zmienność w przesuwaniu optimum
    """)

def run_simulation(params):
    history = []
    fig, ax = plt.subplots()
    env = Environment(alpha_init=torch.tensor([0.0] * params['n_genes']), c=torch.tensor([params['opt_drift']]*params['n_genes']), delta=params['opt_noise'])
    pop = Population(size=params['n_organisms'], n_dim=params['n_genes'])

    for generation in range(params['steps']):
        # 1. Mutacja
        mutate_population(pop, mu=params['mutation_rate'], mu_c=params['gene_mutation_rate'], xi=params['mutation_mag'])

        # 2. Selekcja
        pop.set_individuals(selection(pop, env.get_optimal_phenotype(), params['selection'], params['n_organisms'], params['threshold']))

        # 3. Reprodukcja (w przykładzie jest już wbudowana w selekcję)
        # 4. Zmiana środowiska
        env.update()

        if generation % params['log_interval'] == 0:
            phenotypes = pop.get_individuals().numpy(force=True)
            opt_phenotype = env.get_optimal_phenotype()[:2] # placeholder, bo opt i tak ma mieć 2 wymiary
            history.append((phenotypes, opt_phenotype))

    def animate(i):
        ax.clear()
        data, opt = history[i]
        ax.scatter(data[:, 0], data[:, 1], s=30)
        ax.scatter(opt[0], opt[1], c='red', marker='X', s=100)
        ax.set_title(f"Generation {i*params['log_interval']}")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return ax,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(history), interval=100)
    plt.close()
    return anim.to_html5_video()


if __name__ == "__main__":
    main()
