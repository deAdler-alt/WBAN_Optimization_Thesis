import numpy as np
from src.physics import WBANPhysics
from src.body_model import BodyModel, LANDMARKS

# ==================================================================================
# DEFINICJA PROBLEMU OPTYMALIZACYJNEGO
# Scenariusz: Mamy stałe sensory (np. na kostce, nadgarstku) i szukamy
# optymalnych pozycji dla K węzłów przekaźnikowych (Relays/Cluster Heads).
# ==================================================================================

# 1. Definicja stałych elementów sieci (Źródła danych)
FIXED_SENSORS = [
    {'name': 'ECG_Monitor',  'pos': LANDMARKS['CHEST'],   'data_rate': 500}, # ECG wysyła dużo danych
    {'name': 'Activity_L',   'pos': LANDMARKS['WRIST_L'], 'data_rate': 100}, # Akcelerometr ręka
    {'name': 'Activity_Leg', 'pos': LANDMARKS['ANKLE_L'], 'data_rate': 100}, # Akcelerometr noga
    {'name': 'Temp_Sensor',  'pos': LANDMARKS['BACK'],    'data_rate': 10}   # Temperatura plecy
]

# Hub (Odbiornik końcowy) - zazwyczaj na pasie/pępku
HUB_POS = LANDMARKS['NAVEL']

# Kary (Penalty) - żeby algorytm "bolało", gdy robi głupoty
PENALTY_OFF_BODY = 1.0  # Kara za wyjście poza ciało (Joule) - to dużo w skali mikroJouli!
PENALTY_DISCONNECTED = 0.5

class WBANOptimizationProblem:
    """
    Klasa pomocnicza (Wrapper) dla biblioteki Mealpy.
    """
    
    def __init__(self, n_relays=2):
        self.n_relays = n_relays
        # Wymiar problemu: Każdy relay ma (x, y), więc 2 * n_relays zmiennych
        self.problem_size = 2 * n_relays
        
        # Granice poszukiwań (Bounds) - Cały obszar mapy ciała [cm]
        # X: 0..100, Y: 0..180 (z zapasem)
        self.lb = [0.0] * self.problem_size
        self.ub = [100.0, 180.0] * n_relays
        
        # Konfiguracja dla Mealpy
        self.minmax = "min" # Minimalizujemy energię
        self.log_to = None  # Opcjonalnie: plik logów

    def decode_solution(self, solution_vector):
        """
        Zamienia płaski wektor z Mealpy [x1, y1, x2, y2...] na listę punktów [(x1,y1), (x2,y2)...]
        """
        relays = []
        for i in range(0, len(solution_vector), 2):
            x = solution_vector[i]
            y = solution_vector[i+1]
            relays.append(np.array([x, y]))
        return relays

    def fitness_function(self, solution_vector):
        """
        Główna Funkcja Celu (Fitness Function).
        Oblicza całkowitą energię zużytą przez sieć w jednym cyklu.
        """
        relays = self.decode_solution(solution_vector)
        total_energy = 0.0
        
        # 1. Sprawdź czy Relay'e są na ciele (Constraints)
        for r_pos in relays:
            is_valid = BodyModel.is_valid_position(r_pos[0], r_pos[1])
            if not is_valid:
                # Jeśli Relay jest w powietrzu -> Gigantyczna kara
                return PENALTY_OFF_BODY 
        
        # 2. Symulacja przesyłu danych (Sensor -> Relay/Hub -> Hub)
        # Każdy sensor szuka "najtańszej" drogi do Huba
        for sensor in FIXED_SENSORS:
            s_pos = np.array(sensor['pos'])
            s_zone, _ = BodyModel.get_zone_info(s_pos[0], s_pos[1])
            
            # Opcja A: Transmisja bezpośrednia (Direct -> Hub)
            # Uwaga: Hub jest w strefie NAVEL, która fizycznie jest blisko TORSO_FRONT
            # Używamy typu strefy sensora, żeby określić warunki nadawania
            cost_direct = WBANPhysics.calculate_energy_consumption(
                s_pos, np.array(HUB_POS), location_type=s_zone[1] if s_zone else 'General'
            )
            
            best_cost = cost_direct
            # best_path = 'Direct' # Debugging info
            
            # Opcja B: Transmisja przez Relay (Sensor -> Relay -> Hub)
            # To jest uproszczony routing: 1 hop (Sensor -> Relay) + 1 hop (Relay -> Hub)
            for r_pos in relays:
                # Koszt 1: Sensor -> Relay
                # Typ propagacji zależy od tego, gdzie jest sensor
                cost_hop1 = WBANPhysics.calculate_energy_consumption(
                    s_pos, r_pos, location_type=s_zone[1] if s_zone else 'General'
                )
                
                # Koszt 2: Relay -> Hub
                # Typ propagacji zależy od tego, gdzie jest Relay
                r_zone, r_type = BodyModel.get_zone_info(r_pos[0], r_pos[1])
                cost_hop2 = WBANPhysics.calculate_energy_consumption(
                    r_pos, np.array(HUB_POS), location_type=r_type if r_zone else 'General'
                )
                
                total_path_cost = cost_hop1 + cost_hop2
                
                if total_path_cost < best_cost:
                    best_cost = total_path_cost
                    # best_path = 'Relay'
            
            # Dodaj koszt najlepszej ścieżki (razy ilość danych)
            # Zakładamy, że 1 unit energii to 1 bit, więc mnożymy przez data_rate (ilość pakietów)
            total_energy += best_cost * sensor['data_rate']

        return total_energy

# --- TEST WERYFIKACYJNY ---
if __name__ == "__main__":
    print("--- Fitness Function Test ---")
    
    # Utwórz problem z 1 przekaźnikiem (Relay)
    problem = WBANOptimizationProblem(n_relays=1)
    
    # Scenariusz 1: Relay w złym miejscu (poza ciałem)
    bad_sol = [150.0, 150.0] 
    fit_bad = problem.fitness_function(bad_sol)
    print(f"Bad Solution Fitness (Off-body): {fit_bad} (Oczekiwana kara)")
    
    # Scenariusz 2: Relay w "logicznym" miejscu (np. pas/klatka, blisko huba i sensorów)
    # Punkt (40, 50) to okolice pępka/klatki
    good_sol = [40.0, 50.0]
    fit_good = problem.fitness_function(good_sol)
    print(f"Good Solution Fitness (On-body): {fit_good:.6f} J")
    
    if fit_bad > fit_good:
        print(">> SUKCES: Funkcja celu poprawnie karze złe rozwiązania.")
    else:
        print(">> BŁĄD: Złe rozwiązanie jest oceniane lepiej niż dobre!")