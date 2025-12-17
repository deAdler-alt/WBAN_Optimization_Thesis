import numpy as np
from src.physics import WBANPhysics
from src.body_model import BodyModel, LANDMARKS

# ==================================================================================
# DEFINICJA PROBLEMU OPTYMALIZACYJNEGO (WIELOKRYTERIALNA - POPRAWIONA)
# ==================================================================================

# 1. Definicja stałych elementów sieci
FIXED_SENSORS = [
    {'name': 'ECG_Monitor',  'pos': LANDMARKS['CHEST'],   'data_rate': 500}, 
    {'name': 'Activity_L',   'pos': LANDMARKS['WRIST_L'], 'data_rate': 100}, 
    {'name': 'Activity_Leg', 'pos': LANDMARKS['ANKLE_L'], 'data_rate': 100}, 
    {'name': 'Temp_Sensor',  'pos': LANDMARKS['BACK'],    'data_rate': 10}   
]

HUB_POS = LANDMARKS['NAVEL']

# Kary muszą być MIAŻDŻĄCE w porównaniu do normalnego kosztu
PENALTY_OFF_BODY = 1000.0      
PENALTY_DISCONNECTED = 500.0

# Wagi (Priorytety promotora)
WEIGHTS = {
    'energy': 0.6,      # Najważniejsza jest bateria
    'delay': 0.1,       
    'quality': 0.2,     # Ważna jakość sygnału
    'load': 0.1         
}

# POPRAWIONE Współczynniki normalizacyjne (Calibration)
# Musimy sprowadzić każdą metrykę do zakresu 0.0 - 1.0 dla typowego, poprawnego rozwiązania
NORM_FACTORS = {
    'energy': 0.005,   # Zwiększamy mianownik (by wynik był mniejszy), bo suma energii całej sieci to np. ~0.001-0.003 J
    'delay': 0.1,      # Sekundy
    'quality': 50.0,   # dB (Margines 50dB to luksus, 0dB to tragedia)
    'load': 1.0        # Odchylenie standardowe liczby połączeń
}

class WBANOptimizationProblem:
    
    # Dodajemy parametr 'custom_sensors' do konstruktora
    def __init__(self, n_relays=2, custom_sensors=None):
        self.n_relays = n_relays
        self.problem_size = 2 * n_relays
        self.lb = [0.0] * self.problem_size
        self.ub = [100.0, 180.0] * n_relays
        self.minmax = "min"
        self.log_to = None
        
        # Jeśli podano własne sensory (np. do eksperymentu), użyj ich.
        # W przeciwnym razie użyj domyślnych FIXED_SENSORS.
        if custom_sensors is not None:
            self.sensors = custom_sensors
        else:
            self.sensors = FIXED_SENSORS

    def decode_solution(self, solution_vector):
        relays = []
        for i in range(0, len(solution_vector), 2):
            relays.append(np.array([solution_vector[i], solution_vector[i+1]]))
        return relays

    def fitness_function(self, solution_vector):
        """
        Wielokryterialna Funkcja Celu (Weighted Sum Approach).
        """
        relays = self.decode_solution(solution_vector)
        
        # --- 1. Sprawdzenie Ograniczeń (Constraints) ---
        for r_pos in relays:
            if not BodyModel.is_valid_position(r_pos[0], r_pos[1]):
                return PENALTY_OFF_BODY 
        
        total_energy_J = 0.0
        total_delay_s = 0.0
        min_link_margin_dB = 100.0 
        relay_usage = [0] * self.n_relays 
        
        # --- 2. Symulacja ---
        # ZMIANA: Iterujemy po self.sensors zamiast globalnego FIXED_SENSORS
        for sensor in self.sensors: 
            s_pos = np.array(sensor['pos'])
            s_zone, _ = BodyModel.get_zone_info(s_pos[0], s_pos[1])
            
            # Parametry ścieżki domyślnej (Direct to Hub)
            # Uwaga: Jeśli Direct jest niemożliwy (zbyt duży Path Loss), koszt powinien być wysoki
            dist_dir = WBANPhysics.calculate_distance_m(s_pos, np.array(HUB_POS))
            pl_dir = WBANPhysics.calculate_path_loss_dB(dist_dir, s_zone[1] if s_zone else 'General')
            margin_dir = max(0, 96.0 - pl_dir) # Czułość -96dBm
            
            e_direct = WBANPhysics.calculate_energy_consumption(s_pos, np.array(HUB_POS), location_type=s_zone[1] if s_zone else 'General')
            d_direct = (1500 / 1_000_000)
            
            # Inicjalizacja "najlepszej" ścieżki jako Direct
            chosen_energy = e_direct
            chosen_delay = d_direct
            chosen_margin = margin_dir
            chosen_relay_idx = -1 

            # Szukamy czy Relay jest lepszy
            for idx, r_pos in enumerate(relays):
                # Hop 1
                dist_h1 = WBANPhysics.calculate_distance_m(s_pos, r_pos)
                pl_h1 = WBANPhysics.calculate_path_loss_dB(dist_h1, s_zone[1] if s_zone else 'General')
                margin_h1 = max(0, 96.0 - pl_h1)
                e_hop1 = WBANPhysics.calculate_energy_consumption(s_pos, r_pos, location_type=s_zone[1] if s_zone else 'General')
                d_hop1 = (1500 / 1_000_000)

                # Hop 2
                r_zone, r_type = BodyModel.get_zone_info(r_pos[0], r_pos[1])
                dist_h2 = WBANPhysics.calculate_distance_m(r_pos, np.array(HUB_POS))
                pl_h2 = WBANPhysics.calculate_path_loss_dB(dist_h2, r_type if r_zone else 'General')
                margin_h2 = max(0, 96.0 - pl_h2)
                e_hop2 = WBANPhysics.calculate_energy_consumption(r_pos, np.array(HUB_POS), location_type=r_type if r_zone else 'General')
                d_hop2 = (1500 / 1_000_000)

                # Suma Relay
                e_relay_total = e_hop1 + e_hop2
                # Logika wyboru: Jeśli Relay oszczędza energię LUB pozwala na połączenie (gdy Direct ma zerowy margines)
                if e_relay_total < chosen_energy:
                    chosen_energy = e_relay_total
                    chosen_delay = d_hop1 + d_hop2 + 0.005 # Kara za opóźnienie w węźle (5ms)
                    chosen_margin = min(margin_h1, margin_h2)
                    chosen_relay_idx = idx

            # Sumowanie
            total_energy_J += chosen_energy * sensor['data_rate']
            total_delay_s += chosen_delay
            if chosen_margin < min_link_margin_dB:
                min_link_margin_dB = chosen_margin
            
            if chosen_relay_idx != -1:
                relay_usage[chosen_relay_idx] += 1

        # --- 3. Obliczenie Składników Znormalizowanych ---
        
        # Energia (chcemy min) -> im mniej tym lepiej
        f_energy = total_energy_J / NORM_FACTORS['energy']
        
        # Opóźnienie (chcemy min)
        f_delay = total_delay_s / NORM_FACTORS['delay']
        
        # Jakość (chcemy max margines) -> w funkcji kosztu (min) musimy to odwrócić
        # Wzór: (Max_Margin - Current_Margin) / Norm
        # Jeśli margines jest duży (dobry), licznik jest mały (dobry koszt)
        f_quality = (100.0 - min_link_margin_dB) / NORM_FACTORS['quality']
        
        # Load Balancing (chcemy min std)
        f_load = 0.0
        if sum(relay_usage) > 0:
            f_load = np.std(relay_usage) / NORM_FACTORS['load']

        # Ostateczny Fitness
        fitness = (WEIGHTS['energy'] * f_energy +
                   WEIGHTS['delay']  * f_delay +
                   WEIGHTS['quality'] * f_quality +
                   WEIGHTS['load']   * f_load)
                   
        return fitness

    def get_routing_details(self, solution_vector):
        """
        Zwraca szczegóły tras dla danego rozwiązania (do wizualizacji).
        """
        relays = self.decode_solution(solution_vector)
        paths = []
        
        for sensor in FIXED_SENSORS:
            s_pos = np.array(sensor['pos'])
            s_zone, _ = BodyModel.get_zone_info(s_pos[0], s_pos[1])
            
            # Parametry Direct
            e_direct = WBANPhysics.calculate_energy_consumption(s_pos, np.array(HUB_POS), location_type=s_zone[1] if s_zone else 'General')
            
            best_energy = e_direct
            chosen_path = {'from': s_pos, 'to': np.array(HUB_POS), 'type': 'Direct'}
            
            # Sprawdź Relaye
            for idx, r_pos in enumerate(relays):
                e_hop1 = WBANPhysics.calculate_energy_consumption(s_pos, r_pos, location_type=s_zone[1] if s_zone else 'General')
                
                r_zone, r_type = BodyModel.get_zone_info(r_pos[0], r_pos[1])
                e_hop2 = WBANPhysics.calculate_energy_consumption(r_pos, np.array(HUB_POS), location_type=r_type if r_zone else 'General')
                
                if (e_hop1 + e_hop2) < best_energy:
                    best_energy = e_hop1 + e_hop2
                    # Zapisz dwa segmenty trasy
                    chosen_path = [
                        {'from': s_pos, 'to': r_pos, 'type': 'Relay'}, # Hop 1
                        {'from': r_pos, 'to': np.array(HUB_POS), 'type': 'Relay'}  # Hop 2
                    ]

            # Dodaj do listy tras (obsługa listy lub pojedynczego słownika)
            if isinstance(chosen_path, list):
                paths.extend(chosen_path)
            else:
                paths.append(chosen_path)
                
        return paths

# --- TEST WERYFIKACYJNY ---
if __name__ == "__main__":
    print("--- Fitness Function 2.1 (Calibrated) Check ---")
    problem = WBANOptimizationProblem(n_relays=2)
    
    # Rozwiązanie A: Dobre (na ciele)
    sol_A = [45.0, 30.0, 65.0, 40.0] 
    fit_A = problem.fitness_function(sol_A)
    print(f"Rozwiązanie A (Sensowne): {fit_A:.5f}")
    
    # Rozwiązanie B: Złe (poza ciałem)
    sol_B = [150.0, 150.0, 150.0, 150.0]
    fit_B = problem.fitness_function(sol_B)
    print(f"Rozwiązanie B (Poza ciałem): {fit_B:.5f}")
    
    if fit_B > fit_A:
        print(">> SUKCES: Kara jest wyższa niż koszt dobrego rozwiązania.")
    else:
        print(f">> BŁĄD: Kara ({fit_B}) nadal za mała w stosunku do kosztu ({fit_A})!")