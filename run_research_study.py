import numpy as np
import pandas as pd
import time
from mealpy import FloatVar
# Importy algorytmów
from mealpy.evolutionary_based import GA, DE
from mealpy.swarm_based import PSO, GWO

# Importy z naszych modułów
from src.fitness import WBANOptimizationProblem, FIXED_SENSORS
from src.body_model import BodyModel

# ==============================================================================
# 1. KONFIGURACJA EKSPERYMENTU (Zgodna z ustaleniami)
# ==============================================================================
# Scenariusze liczby sensorów (WBAN scale)
SCENARIOS_SENSORS = [6, 8, 10, 12, 15, 20] 

# Liczba powtórzeń dla statystyki (30 to standard naukowy)
N_TRIALS = 30 

# Ustawienia Algorytmów (Sprawiedliwe warunki: 50 epok x 30 osobników = 1500 ocen)
EPOCHS = 50
POP_SIZE = 30
N_RELAYS = 2 # Stała liczba Relayów (szukamy ich miejsca)

# Lista badanych algorytmów
ALGORITHMS = {
    'GA':  GA.BaseGA,       # Genetic Algorithm
    'DE':  DE.OriginalDE,   # Differential Evolution
    'PSO': PSO.OriginalPSO, # Particle Swarm Optimization
    'GWO': GWO.OriginalGWO  # Grey Wolf Optimizer
}

# ==============================================================================
# 2. GENERATOR SENSORÓW
# ==============================================================================
def get_sensor_placement(n_sensors, seed=42):
    """
    Generuje powtarzalny układ sensorów dla danego scenariusza.
    Dzięki seed=42, każdy algorytm dostanie TA SAMĄ mapę ciała do rozwiązania.
    """
    np.random.seed(seed) 
    sensors = []
    # Kopiuj bazowe sensory medyczne (żeby nie modyfikować oryginału)
    base = [s.copy() for s in FIXED_SENSORS[:min(len(FIXED_SENSORS), n_sensors)]]
    sensors.extend(base)
    
    # Dopełnij losowymi na ciele
    while len(sensors) < n_sensors:
        pos = BodyModel.get_random_valid_position()
        sensors.append({'name': f'S_{len(sensors)}', 'pos': pos, 'data_rate': 100})
    
    return sensors

# ==============================================================================
# 3. SILNIK EKSPERYMENTU
# ==============================================================================
def run_full_study():
    print(f"--- ROZPOCZYNAM BADANIE WBAN ---")
    print(f"Scenariusze (Sensory): {SCENARIOS_SENSORS}")
    print(f"Algorytmy: {list(ALGORITHMS.keys())}")
    print(f"Konfiguracja: {N_TRIALS} powtórzeń, {EPOCHS} epok, {POP_SIZE} populacji")
    print("-" * 60)

    results_db = []
    
    # Pętla po liczbie sensorów (Skalowalność)
    for n_nodes in SCENARIOS_SENSORS:
        print(f"\n>>> PRZETWARZANIE SCENARIUSZA: {n_nodes} SENSORÓW")
        
        # Generujemy układ ciała (stały dla wszystkich algorytmów w tym scenariuszu)
        # Używamy seed=n_nodes, żeby dla 6 sensorów zawsze był ten sam układ
        current_sensors = get_sensor_placement(n_nodes, seed=n_nodes)
        
        # Inicjalizacja problemu
        problem = WBANOptimizationProblem(n_relays=N_RELAYS, custom_sensors=current_sensors)
        problem_dict = {
            "obj_func": problem.fitness_function,
            "bounds": FloatVar(lb=problem.lb, ub=problem.ub),
            "minmax": "min",
            "log_to": None
        }

        # Pętla po algorytmach
        for algo_name, algo_class in ALGORITHMS.items():
            print(f"   [Algorytm: {algo_name}] Trwa {N_TRIALS} uruchomień...", end="", flush=True)
            
            start_time_algo = time.time()
            
            # Pętla Monte Carlo (Statystyka)
            for i in range(N_TRIALS):
                # Inicjalizacja modelu
                model = algo_class(epoch=EPOCHS, pop_size=POP_SIZE)
                
                # Rozwiązanie (z mierzonym czasem CPU)
                t0 = time.time()
                res = model.solve(problem_dict)
                t_exec = time.time() - t0
                
                best_sol = res.solution
                best_fitness = res.target.fitness
                
                # --- EKSTRAKCJA METRYK FIZYCZNYCH ---
                # To jest kluczowe: Zamieniamy "Fitness" na Fizykę
                metrics = problem.get_metrics_details(best_sol)
                
                # Zapis do bazy
                results_db.append({
                    'Scenario_Sensors': n_nodes,
                    'Algorithm': algo_name,
                    'Trial_ID': i + 1,
                    'Fitness_Cost': best_fitness,
                    'Execution_Time_s': t_exec,
                    # Fizyczne Metryki
                    'Energy_Total_J': metrics['Energy'],
                    'Avg_Delay_s': metrics['Delay'] / n_nodes, # Średnie opóźnienie na pakiet
                    'Min_Link_Margin_dB': metrics['Quality'],
                    'Network_Load_Std': metrics.get('Load_Std', 0.0) # Jeśli dodasz to do get_metrics_details
                })
            
            duration = time.time() - start_time_algo
            print(f" Zakończono w {duration:.1f}s")

    # ==========================================================================
    # 4. EKSPORT WYNIKÓW
    # ==========================================================================
    df = pd.DataFrame(results_db)
    
    # Oblicz statystyki zbiorcze (średnie)
    summary = df.groupby(['Scenario_Sensors', 'Algorithm'])[['Energy_Total_J', 'Avg_Delay_s', 'Execution_Time_s']].mean()
    
    print("\n" + "="*60)
    print("PODSUMOWANIE WYNIKÓW (Średnie wartości):")
    print("="*60)
    print(summary)
    
    # Zapis do pliku
    filename = "WBAN_Experiment_Results.csv"
    df.to_csv(filename, index=False)
    print(f"\n[SUKCES] Pełne dane surowe zapisano do: {filename}")
    print("Gotowe do analizy i tworzenia wykresów w następnym etapie.")

if __name__ == "__main__":
    run_full_study()