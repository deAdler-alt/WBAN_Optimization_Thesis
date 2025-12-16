from src.utils import plot_body_simulation
import numpy as np
import pandas as pd
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO
from mealpy import FloatVar

# Importy z naszych modułów
from src.fitness import WBANOptimizationProblem
from src.body_model import BodyModel, LANDMARKS
from src.physics import WBANPhysics

# ==================================================================================
# KONFIGURACJA EKSPERYMENTU
# ==================================================================================
N_RELAYS = 2         # Szukamy 2 optymalnych miejsc dla przekaźników
EPOCHS = 50          # Liczba iteracji (pokoleń)
POP_SIZE = 30        # Liczba "osobników" w populacji

def run_experiment():
    print(f"--- ROZPOCZYNAM SYMULACJĘ WBAN OPTIMIZATION ---")
    print(f"Cel: Znaleźć optymalne pozycje dla {N_RELAYS} węzłów Relay.")
    
    # 1. Inicjalizacja Problemu (Nasza klasa Wrapper)
    wban_problem = WBANOptimizationProblem(n_relays=N_RELAYS)
    
    # 2. Definicja słownika problemu dla Mealpy (Standard v3.0+)
    problem_dict = {
        "obj_func": wban_problem.fitness_function,
        "bounds": FloatVar(lb=wban_problem.lb, ub=wban_problem.ub),
        "minmax": wban_problem.minmax,
        "log_to": None,
    }

    # ==============================================================================
    # ALGORYTM 1: GA (Genetic Algorithm) - Klasyka
    # ==============================================================================
    print("\n>>> Uruchamiam GA (Genetic Algorithm)...")
    model_ga = GA.BaseGA(epoch=EPOCHS, pop_size=POP_SIZE)
    
    # Rozwiązywanie (Mealpy 3.0+ zwraca obiekt Agent)
    result_ga = model_ga.solve(problem_dict)
    
    # Wyciągamy dane z obiektu
    best_position_ga = result_ga.solution
    best_fitness_ga = result_ga.target.fitness
    
    print(f"GA Zakończony. Najlepsza energia: {best_fitness_ga:.9f} J")

    # ==============================================================================
    # ALGORYTM 2: PSO (Particle Swarm Optimization) - Inteligencja Roju
    # ==============================================================================
    print("\n>>> Uruchamiam PSO (Particle Swarm)...")
    model_pso = PSO.OriginalPSO(epoch=EPOCHS, pop_size=POP_SIZE)
    
    # Rozwiązywanie
    result_pso = model_pso.solve(problem_dict)
    
    # Wyciągamy dane z obiektu
    best_position_pso = result_pso.solution
    best_fitness_pso = result_pso.target.fitness
    
    print(f"PSO Zakończony. Najlepsza energia: {best_fitness_pso:.9f} J")

    # ==============================================================================
    # ANALIZA WYNIKÓW
    # ==============================================================================
    print("\n" + "="*50)
    print("PODSUMOWANIE WYNIKÓW")
    print("="*50)
    
    # Wybierz zwycięzcę
    if best_fitness_ga < best_fitness_pso:
        winner = "GA"
        best_sol = best_position_ga
        best_val = best_fitness_ga
    else:
        winner = "PSO"
        best_sol = best_position_pso
        best_val = best_fitness_pso
        
    print(f"ZWYCIĘZCA: {winner} (Koszt: {best_val:.9f} J)")
    
    # Dekodowanie rozwiązania (Gdzie on te sensory postawił?)
    decoded_relays = wban_problem.decode_solution(best_sol)
    
    print("\nOPTYMALNE ROZMIESZCZENIE PRZEKAŹNIKÓW:")
    hub = BodyModel.get_hub_position()
    
    for i, r_pos in enumerate(decoded_relays):
        zone, ptype = BodyModel.get_zone_info(r_pos[0], r_pos[1])
        dist_hub = WBANPhysics.calculate_distance_m(r_pos, hub)
        print(f"  Relay {i+1}: X={r_pos[0]:.1f}, Y={r_pos[1]:.1f} cm")
        print(f"     -> Strefa: {zone} ({ptype})")
        print(f"     -> Odległość do Huba: {dist_hub*100:.1f} cm")

    plot_body_simulation(decoded_relays, title=f"Wynik Optymalizacji ({winner})")

if __name__ == "__main__":
    run_experiment()