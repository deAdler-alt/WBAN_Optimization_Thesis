import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO
from mealpy import FloatVar

from src.fitness import WBANOptimizationProblem, FIXED_SENSORS
from src.body_model import BodyModel

# --- KONFIGURACJA STYLU IEEE (Formalny, czarno-biały/szary) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'text.usetex': False # Jeśli masz LaTeX zainstalowany w systemie, zmień na True dla ładniejszych wzorów
})

def generate_random_sensors(n_sensors):
    """Generuje N losowych sensorów na ciele (do testów skalowalności)."""
    sensors = []
    # Zawsze dodaj ECG i Hub-area (baza)
    sensors.append(FIXED_SENSORS[0]) # ECG
    
    for i in range(n_sensors - 1):
        pos = BodyModel.get_random_valid_position()
        sensors.append({
            'name': f'Rand_Sensor_{i}',
            'pos': pos,
            'data_rate': 100 # Standardowe dane
        })
    return sensors

def run_convergence_comparison():
    """EKSPERYMENT 1: Porównanie zbieżności GA vs PSO"""
    print(">>> [Exp 1] Uruchamianie analizy zbieżności (GA vs PSO)...")
    
    problem = WBANOptimizationProblem(n_relays=2)
    problem_dict = {
        "obj_func": problem.fitness_function,
        "bounds": FloatVar(lb=problem.lb, ub=problem.ub),
        "minmax": "min",
        "log_to": None
    }
    
    EPOCHS = 50
    POP_SIZE = 30
    
    # 1. PSO
    model_pso = PSO.OriginalPSO(epoch=EPOCHS, pop_size=POP_SIZE)
    model_pso.solve(problem_dict)
    fit_pso = model_pso.history.list_global_best_fit
    
    # 2. GA
    model_ga = GA.BaseGA(epoch=EPOCHS, pop_size=POP_SIZE)
    model_ga.solve(problem_dict)
    fit_ga = model_ga.history.list_global_best_fit
    
    # Rysowanie Wykresu
    epochs = range(1, len(fit_pso) + 1)
    plt.figure(figsize=(7, 5))
    
    plt.plot(epochs, fit_pso, marker='o', markevery=5, linestyle='-', color='black', label='PSO')
    plt.plot(epochs, fit_ga, marker='s', markevery=5, linestyle='--', color='gray', label='GA')
    
    plt.xlabel('Iteration (Epoch)')
    plt.ylabel('Cost Function Value (Fitness)')
    plt.title('Convergence Analysis: GA vs PSO')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('IEEE_Exp1_Convergence.png')
    print("   -> Zapisano: IEEE_Exp1_Convergence.png")

def run_scalability_test():
    """EKSPERYMENT 2: Zużycie energii vs Liczba sensorów"""
    print("\n>>> [Exp 2] Uruchamianie testu skalowalności (Energy vs Sensors)...")
    
    sensor_counts = [4, 8, 12, 16, 20]
    results_pso = []
    results_ga = []
    
    for n in sensor_counts:
        print(f"   -> Symulacja dla {n} sensorów...")
        
        # Generuj losowe sensory (taki sam zestaw dla obu algorytmów)
        current_sensors = generate_random_sensors(n)
        problem = WBANOptimizationProblem(n_relays=2, custom_sensors=current_sensors)
        
        problem_dict = {
            "obj_func": problem.fitness_function,
            "bounds": FloatVar(lb=problem.lb, ub=problem.ub),
            "minmax": "min",
            "log_to": None
        }
        
        # PSO
        model_pso = PSO.OriginalPSO(epoch=30, pop_size=20) # Mniej epok dla szybkości
        res_pso = model_pso.solve(problem_dict)
        results_pso.append(res_pso.target.fitness)
        
        # GA
        model_ga = GA.BaseGA(epoch=30, pop_size=20)
        res_ga = model_ga.solve(problem_dict)
        results_ga.append(res_ga.target.fitness)
    
    # Rysowanie Wykresu
    plt.figure(figsize=(7, 5))
    
    plt.plot(sensor_counts, results_pso, marker='^', linestyle='-', color='black', label='PSO')
    plt.plot(sensor_counts, results_ga, marker='D', linestyle='--', color='gray', label='GA')
    
    plt.xlabel('Number of Sensors')
    plt.ylabel('Total Network Cost (Weighted Fitness)')
    plt.title('Scalability Analysis: Impact of Sensor Density')
    plt.xticks(sensor_counts)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('IEEE_Exp2_Scalability.png')
    print("   -> Zapisano: IEEE_Exp2_Scalability.png")

if __name__ == "__main__":
    run_convergence_comparison()
    run_scalability_test()