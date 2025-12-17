import numpy as np
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO
from mealpy import FloatVar

from src.fitness import WBANOptimizationProblem
from src.body_model import BodyModel
from src.utils import plot_body_simulation, plot_convergence

# KONFIGURACJA
N_RELAYS = 2
EPOCHS = 50
POP_SIZE = 30

def run_experiment():
    print(f"--- ANALIZA TECHNICZNA WBAN ---")
    
    # 1. Definicja Problemu
    wban_problem = WBANOptimizationProblem(n_relays=N_RELAYS)
    
    problem_dict = {
        "obj_func": wban_problem.fitness_function,
        "bounds": FloatVar(lb=wban_problem.lb, ub=wban_problem.ub),
        "minmax": wban_problem.minmax,
        "log_to": None,
    }

    # 2. Uruchomienie PSO (Jest zazwyczaj szybsze i stabilniejsze dla ciągłych problemów)
    print("\n>>> Uruchamiam PSO...")
    model_pso = PSO.OriginalPSO(epoch=EPOCHS, pop_size=POP_SIZE)
    result_pso = model_pso.solve(problem_dict)
    
    best_sol = result_pso.solution
    best_fit = result_pso.target.fitness
    
    print(f"Najlepszy koszt: {best_fit:.5f}")

    # 3. GENEROWANIE RAPORTU TECHNICZNEGO
    
    # A. Wykres Zbieżności (Dowód działania procesu)
    plot_convergence(model_pso.history, algorithm_name="PSO")
    
    # B. Wyodrębnienie Tras (Routing)
    active_paths = wban_problem.get_routing_details(best_sol)
    
    # C. Mapa z połączeniami
    relays = wban_problem.decode_solution(best_sol)
    plot_body_simulation(relays, active_paths=active_paths, title="Optymalna Topologia Sieci (PSO)")
    
    print("\n--- ANALIZA ROZMIESZCZENIA ---")
    for i, r in enumerate(relays):
        zone, _ = BodyModel.get_zone_info(r[0], r[1])
        print(f"Relay {i+1}: ({r[0]:.1f}, {r[1]:.1f}) -> {zone}")

if __name__ == "__main__":
    run_experiment()