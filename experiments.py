import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mealpy import FloatVar

# --- IMPORTY ALGORYTMÓW (2 Ewolucyjne + 2 Rojowe) ---
from mealpy.evolutionary_based import GA, DE    # Genetyczny + Różnicowy
from mealpy.swarm_based import PSO, GWO         # Cząsteczki + Szare Wilki

from src.fitness import WBANOptimizationProblem, FIXED_SENSORS
from src.body_model import BodyModel

# ==============================================================================
# KONFIGURACJA STYLU IEEE (Publikacja Naukowa)
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# Konfiguracja wizualna dla 4 algorytmów (Czarno-białe/Szare - bezpieczne do druku)
ALGO_CONFIG = {
    # Grupa Ewolucyjna
    'GA':  {'color': 'black', 'marker': 's', 'linestyle': '--', 'label': 'GA (Genetic)'},      # Kwadrat, przerywana
    'DE':  {'color': 'gray',  'marker': 'D', 'linestyle': '--', 'label': 'DE (Differential)'}, # Romb, przerywana
    
    # Grupa Rojowa
    'PSO': {'color': 'black', 'marker': '^', 'linestyle': '-',  'label': 'PSO (Particle)'},    # Trójkąt, ciągła
    'GWO': {'color': 'gray',  'marker': 'o', 'linestyle': '-',  'label': 'GWO (Grey Wolf)'}    # Kółko, ciągła
}

def generate_dynamic_sensors(n_sensors):
    """Generuje losowy zestaw sensorów (zachowując bazowe medyczne)."""
    sensors = []
    base_sensors = FIXED_SENSORS[:min(len(FIXED_SENSORS), n_sensors)]
    sensors.extend(base_sensors)
    
    while len(sensors) < n_sensors:
        pos = BodyModel.get_random_valid_position()
        sensors.append({'name': f'S_{len(sensors)+1}', 'pos': pos, 'data_rate': 100})
    return sensors

def run_experiment_convergence():
    """
    EXP 1: Analiza Zbieżności (Szybkość uczenia się)
    """
    print("\n>>> [EXP 1] Analiza Zbieżności (GA/DE vs PSO/GWO)...")
    
    # Parametry symulacji
    EPOCHS = 60
    POP_SIZE = 30
    N_RELAYS = 2
    
    problem = WBANOptimizationProblem(n_relays=N_RELAYS)
    problem_dict = {
        "obj_func": problem.fitness_function,
        "bounds": FloatVar(lb=problem.lb, ub=problem.ub),
        "minmax": "min",
        "log_to": None
    }
    
    # Słownik na modele
    models = {
        'GA':  GA.BaseGA(epoch=EPOCHS, pop_size=POP_SIZE),
        'DE':  DE.OriginalDE(epoch=EPOCHS, pop_size=POP_SIZE), # Differential Evolution
        'PSO': PSO.OriginalPSO(epoch=EPOCHS, pop_size=POP_SIZE),
        'GWO': GWO.OriginalGWO(epoch=EPOCHS, pop_size=POP_SIZE) # Grey Wolf Optimizer
    }
    
    # Uruchamianie pętli
    for name, model in models.items():
        print(f"    Running {name}...")
        model.solve(problem_dict)

    # Rysowanie
    plt.figure(figsize=(8, 6))
    x_epochs = range(1, EPOCHS + 1)
    
    for name, model in models.items():
        cfg = ALGO_CONFIG[name]
        # Wybieramy global best fitness z historii
        y_history = model.history.list_global_best_fit
        
        plt.plot(x_epochs, y_history, 
                 color=cfg['color'], marker=cfg['marker'], linestyle=cfg['linestyle'], 
                 label=cfg['label'], markevery=5) # Markery co 5, żeby nie zamazać
    
    plt.xlabel('Iterations (Epochs)')
    plt.ylabel('Cost Function Value (Fitness)')
    plt.title('Convergence Analysis: Evolutionary vs Swarm')
    plt.legend()
    plt.tight_layout()
    
    filename = 'FIG_1_Convergence_4Algos.png'
    plt.savefig(filename)
    print(f"    [SAVED] {filename}")

def run_experiment_scalability():
    """
    EXP 2: Analiza Skalowalności (Wpływ liczby sensorów)
    """
    print("\n>>> [EXP 2] Analiza Skalowalności (Energy vs Sensors)...")
    
    sensor_counts = [5, 10, 15, 20]
    results = {name: [] for name in ALGO_CONFIG.keys()}
    
    # Lżejsze ustawienia do pętli
    EPOCHS = 30
    POP_SIZE = 20
    N_RELAYS = 2
    
    for n in sensor_counts:
        print(f"    Simulating network size: {n} nodes...")
        current_sensors = generate_dynamic_sensors(n)
        
        # Tworzymy problem z nowym zestawem sensorów
        problem = WBANOptimizationProblem(n_relays=N_RELAYS, custom_sensors=current_sensors)
        problem_dict = {
            "obj_func": problem.fitness_function,
            "bounds": FloatVar(lb=problem.lb, ub=problem.ub),
            "minmax": "min",
            "log_to": None
        }
        
        # Definicja modeli wewnątrz pętli (muszą być świeże dla każdego n)
        models = {
            'GA':  GA.BaseGA(epoch=EPOCHS, pop_size=POP_SIZE),
            'DE':  DE.OriginalDE(epoch=EPOCHS, pop_size=POP_SIZE),
            'PSO': PSO.OriginalPSO(epoch=EPOCHS, pop_size=POP_SIZE),
            'GWO': GWO.OriginalGWO(epoch=EPOCHS, pop_size=POP_SIZE)
        }
        
        for name, model in models.items():
            res = model.solve(problem_dict)
            results[name].append(res.target.fitness)

    # Rysowanie
    plt.figure(figsize=(8, 6))
    
    for name, vals in results.items():
        cfg = ALGO_CONFIG[name]
        plt.plot(sensor_counts, vals, 
                 color=cfg['color'], marker=cfg['marker'], linestyle=cfg['linestyle'], 
                 label=cfg['label'])
    
    plt.xlabel('Number of Sensor Nodes')
    plt.ylabel('Total Network Cost (Normalized)')
    plt.title('Scalability Analysis: Network Size Impact')
    plt.xticks(sensor_counts)
    plt.legend()
    plt.tight_layout()
    
    filename = 'FIG_2_Scalability_4Algos.png'
    plt.savefig(filename)
    print(f"    [SAVED] {filename}")

if __name__ == "__main__":
    run_experiment_convergence()
    run_experiment_scalability()