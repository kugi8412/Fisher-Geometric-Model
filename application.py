import streamlit as st
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import streamlit.components.v1 as components

class Allele:
    def __init__(self, value=None):
        if value is None:
            self.value = torch.rand(1).item()
        else:
            self.value = max(0.0, min(1.0, value))
        
    @property
    def type(self):
        return 'A' if self.value >= 0.5 else 'a'
    
    def mutate(self, magnitude):
        self.value = max(0.0, min(1.0, self.value + torch.normal(0.0, magnitude).item()))
    
    def __repr__(self):
        return f"{self.type}({self.value:.2f})"

class Gene:
    def __init__(self):
        self.alleles = [Allele(), Allele()]
    
    @property
    def value(self):
        return sum(a.value for a in self.alleles)
    
    @property
    def representation(self):
        types = [a.type for a in self.alleles]
        return f"{types[0]}{types[1]}"
    
    def mutate(self, magnitude):
        for allele in self.alleles:
            if torch.rand(1) < 0.5:
                allele.mutate(magnitude)

class Organism:
    def __init__(self, n_genes=10, genes=None):
        if genes:
            self.genes = genes
        else:
            self.genes = [Gene() for _ in range(n_genes)]
        self.phenotype = self.get_phenotype()
        self.fitness = 0.0
    
    def get_phenotype(self):
        return torch.tensor([g.value for g in self.genes], dtype=torch.float32)
    
    def update_fitness(self, optimal, s):
        distance = torch.norm(self.phenotype - optimal)
        self.fitness = torch.exp(-distance**2 / (2*s)).item()
        return self.fitness
    
    def reproduce(self, rate, mutation_rate, mutation_mag):
        if self.fitness > (1 - rate):
            new_genes = []
            for gene in self.genes:
                new_gene = Gene()
                new_gene.alleles = [Allele(a.value) for a in gene.alleles]
                new_genes.append(new_gene)
            
            offspring = Organism(genes=new_genes)
            if torch.rand(1) < mutation_rate:
                for gene in offspring.genes:
                    gene.mutate(mutation_mag)
            offspring.phenotype = offspring.get_phenotype()
            return offspring

class FisherEnvironment:
    def __init__(self, params):
        self.params = params
        self.organisms = [Organism(params['n_genes']) for _ in range(params['n_organisms'])]
        self.optimal = torch.zeros(params['n_genes'])
        
    def step(self):
        # Mutacja
        for org in self.organisms:
            if torch.rand(1) < self.params['mutation_rate']:
                for gene in org.genes:
                    gene.mutate(self.params['mutation_mag'])
        
        # Selekcja
        for org in self.organisms:
            org.update_fitness(self.optimal, self.params['selection'])
        self.organisms.sort(key=lambda x: -x.fitness)
        self.organisms = self.organisms[:self.params['n_organisms']]
        
        # Rozmnażanie
        new_organisms = []
        for org in self.organisms:
            offspring = org.reproduce(self.params['reproduction_rate'],
                                     self.params['mutation_rate'],
                                     self.params['mutation_mag'])
            if offspring:
                new_organisms.append(offspring)
        self.organisms.extend(new_organisms)
        
        # Aktualizacja optimum
        self.optimal += torch.normal(self.params['opt_drift'], 
                                   self.params['opt_noise'], 
                                   size=self.optimal.shape)

def run_simulation(params):
    env = FisherEnvironment(params)
    fig, ax = plt.subplots()
    history = []
    
    for _ in range(params['steps']):
        env.step()
        if len(env.organisms) == 0:
            break
        
        if _ % params['log_interval'] == 0:
            phenotypes = np.array([org.phenotype.numpy() for org in env.organisms])
            if phenotypes.shape[0] > 1:
                pca = PCA(n_components=2)
                proj = pca.fit_transform(phenotypes)
                opt_proj = pca.transform(env.optimal.numpy().reshape(1, -1))
                history.append((proj, opt_proj))
    
    def animate(i):
        ax.clear()
        data, opt = history[i]
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5)
        ax.scatter(opt[0, 0], opt[0, 1], c='red', marker='X', s=100)
        ax.set_title(f"Generation {i*params['log_interval']}")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return ax,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(history), interval=100)
    plt.close()
    return anim.to_html5_video()

# Konfiguracja interfejsu Streamlit
st.title("Symulacja Ewolucji Według Modelu Fishera")
st.markdown("""
Symulacja populacji organizmów ewoluujących w kierunku zmieniającego się optimum fenotypowego.
Każdy gen składa się z dwóch alleli (A/a), których wartości wpływają na fenotyp.
""")

params = {
    'n_organisms': st.sidebar.slider("Liczba organizmów", 10, 1000, 200),
    'n_genes': st.sidebar.slider("Liczba genów", 1, 50, 10),
    'selection': st.sidebar.slider("Siła selekcji", 0.1, 10.0, 1.0),
    'mutation_rate': st.sidebar.slider("Częstość mutacji", 0.0, 1.0, 0.1),
    'mutation_mag': st.sidebar.slider("Siła mutacji", 0.0, 1.0, 0.1),
    'reproduction_rate': st.sidebar.slider("Współczynnik reprodukcji", 0.0, 1.0, 0.5),
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