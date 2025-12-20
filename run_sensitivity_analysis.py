import numpy as np
import pandas as pd
import time
from mealpy import FloatVar
# TYLKO 3 ALGORYTMY (Bez DE)
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO, GWO

from src.fitness import WBANOptimizationProblem, FIXED_SENSORS
from src.body_model import BodyModel

# ==============================================================================
# 1. KONFIGURACJA PACZEK (ZASOBÓW)
# ==============================================================================
CONFIG_PACKS = {
    'A_Eco':      {'epoch': 15,  'pop_size': 10},  # Słaby sprzęt
    'B_Standard': {'epoch': 50,  'pop_size': 30},  # Standard
    'C_High':     {'epoch': 100, 'pop_size': 50}   # Mocny sprzęt
}

# Stały, trudny scenariusz
SCENARIO_SENSORS = 15
N_RELAYS = 2
N_TRIALS = 30

ALGORITHMS = {
    'GA':  GA.BaseGA,
    'PSO': PSO.OriginalPSO,
    'GWO': GWO.OriginalGWO
}

# ==============================================================================
# 2. GENERATOR
# ==============================================================================
def get_sensor_placement(n_sensors, seed=42):
    np.random.seed(seed) 
    sensors = []
    base = [s.copy() for s in FIXED_SENSORS[:min(len(FIXED_SENSORS), n_sensors)]]
    sensors.extend(base)
    while len(sensors) < n_sensors:
        pos = BodyModel.get_random_valid_position()
        sensors.append({'name': f'S_{len(sensors)}', 'pos': pos, 'data_rate': 100})
    return sensors

# ==============================================================================
# 3. SILNIK TESTOWY
# ==============================================================================
def run_sensitivity_study():
    print("============================================================")
    print("   ANALIZA WRAŻLIWOŚCI (SENSITIVITY) - GOLD MASTER")
    print("============================================================")
    
    fixed_sensors = get_sensor_placement(SCENARIO_SENSORS, seed=SCENARIO_SENSORS)
    results_db = []

    for pack_name, params in CONFIG_PACKS.items():
        print(f"\n>>> PACZKA: {pack_name} {params}")
        
        problem = WBANOptimizationProblem(n_relays=N_RELAYS, custom_sensors=fixed_sensors)
        problem_dict = {
            "obj_func": problem.fitness_function,
            "bounds": FloatVar(lb=problem.lb, ub=problem.ub),
            "minmax": "min",
            "log_to": None
        }

        for algo_name, algo_class in ALGORITHMS.items():
            print(f"   [{algo_name}] ... ", end="", flush=True)
            
            for i in range(N_TRIALS):
                model = algo_class(epoch=params['epoch'], pop_size=params['pop_size'])
                
                t0 = time.time()
                res = model.solve(problem_dict)
                t_exec = time.time() - t0
                
                # Zapisujemy wynik
                # Fitness < 100 uznajemy za sukces (brak kary 1000)
                is_success = res.target.fitness < 100.0
                
                results_db.append({
                    'Config_Pack': pack_name,
                    'Algorithm': algo_name,
                    'Fitness_Cost': res.target.fitness,
                    'Execution_Time_s': t_exec,
                    'Is_Success': is_success
                })
            print("Gotowe")

    # Zapis
    df = pd.DataFrame(results_db)
    df.to_csv("WBAN_Sensitivity_Results.csv", index=False)
    print("\n[SUKCES] Dane zapisano do: WBAN_Sensitivity_Results.csv")

if __name__ == "__main__":
    run_sensitivity_study()