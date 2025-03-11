# config.py

import numpy as np

# -------------------
# PARAMETRY POPULACJI
# -------------------
N = 50           # liczba osobników w populacji
n = 2            # wymiar przestrzeni fenotypowej

# --------------------
# PARAMETRY MUTACJI
# --------------------
mu = 0.1         # prawdopodobieństwo mutacji dla osobnika
mu_c = 0.5       # prawdopodobieństwo mutacji konkretnej cechy, jeśli osobnik mutuje
xi = 0.1         # odchylenie standardowe w rozkładzie normalnym mutacji

# --------------------
# PARAMETRY SELEKCJI
# --------------------
sigma = 0.2      # parametr w funkcji fitness (kontroluje siłę selekcji)
threshold = 0.1  # przykładowy próg do selekcji progowej (do ewentualnego użycia)

# --------------------
# PARAMETRY ŚRODOWISKA
# --------------------
# Początkowe alpha(t)
alpha0 = np.array([0.0, 0.0])  
# Wektor kierunkowej zmiany c
c = np.array([0.01, 0.01])     
delta = 0.01    # odchylenie standardowe dla fluktuacji
max_generations = 100  # liczba pokoleń do zasymulowania

# ----------------------
# PARAMETRY REPRODUKCJI
# ----------------------
# W wersji bezpłciowej zakładamy klonowanie z uwzględnieniem mutacji.
# Jeśli chcemy modelować płciowo, trzeba dodać odpowiednie parametry.
