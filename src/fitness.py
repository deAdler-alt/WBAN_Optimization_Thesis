import numpy as np
from src.physics import WBANPhysics
from src.body_model import BodyModel, LANDMARKS

# ==================================================================================
# DEFINICJA PROBLEMU OPTYMALIZACYJNEGO (WIELOKRYTERIALNA)
# ==================================================================================

# 1. Definicja stałych elementów sieci
FIXED_SENSORS = [
    {'name': 'ECG_Monitor',  'pos': LANDMARKS['CHEST'],   'data_rate': 500}, 
    {'name': 'Activity_L',   'pos': LANDMARKS['WRIST_L'], 'data_rate': 100}, 
    {'name': 'Activity_Leg', 'pos': LANDMARKS['ANKLE_L'], 'data_rate': 100}, 
    {'name': 'Temp_Sensor',  'pos': LANDMARKS['BACK'],    'data_rate': 10}   
]

HUB_POS = LANDMARKS['NAVEL']

# Kary
PENALTY_OFF_BODY = 1000.0      
PENALTY_DISCONNECTED = 500.0
PENALTY_OVERLAP = 800.0

# Minimalny odstęp między urządzeniami (cm)
MIN_DISTANCE_CM = 10.0 

# Wagi
WEIGHTS = {
    'energy': 0.6,
    'delay': 0.1,       
    'quality': 0.2,
    'load': 0.1         
}

# Kalibracja normalizacji
NORM_FACTORS = {
    'energy': 0.005,
    'delay': 0.1,
    'quality': 50.0,
    'load': 1.0
}

class WBANOptimizationProblem:
    
    def __init__(self, n_relays=2, custom_sensors=None):
        self.n_relays = n_relays
        self.problem_size = 2 * n_relays
        self.lb = [0.0] * self.problem_size
        self.ub = [100.0, 180.0] * n_relays
        self.minmax = "min"
        self.log_to = None
        
        if custom_sensors is not None:
            self.sensors = custom_sensors
        else:
            self.sensors = FIXED_SENSORS

    def decode_solution(self, solution_vector):
        relays = []
        for i in range(0, len(solution_vector), 2):
            relays.append(np.array([solution_vector[i], solution_vector[i+1]]))
        return relays

    def check_overlap(self, relays):
        """
        Sprawdza, czy sensory na siebie nie wchodzą.
        Zwraca True, jeśli jest kolizja (overlap).
        """
        # 1. Sprawdź Relay <-> Relay
        for i in range(len(relays)):
            for j in range(i + 1, len(relays)):
                dist = np.linalg.norm(relays[i] - relays[j])
                if dist < MIN_DISTANCE_CM:
                    return True # Kolizja między relayami
        
        # 2. Sprawdź Relay <-> Fixed Sensor
        for r_pos in relays:
            for s in self.sensors:
                s_pos = np.array(s['pos'])
                dist = np.linalg.norm(r_pos - s_pos)
                if dist < MIN_DISTANCE_CM:
                    return True # Kolizja z sensorem medycznym
            
            # 3. Sprawdź Relay <-> Hub
            dist_hub = np.linalg.norm(r_pos - np.array(HUB_POS))
            if dist_hub < MIN_DISTANCE_CM:
                return True # Kolizja z Hubem
                
        return False

    def fitness_function(self, solution_vector):
        relays = self.decode_solution(solution_vector)
        
        # --- 1. Sprawdzenie Ograniczeń (Constraints) ---
        
        # A. Czy na ciele?
        for r_pos in relays:
            if not BodyModel.is_valid_position(r_pos[0], r_pos[1]):
                return PENALTY_OFF_BODY
        
        # B. Czy nie ma kolizji?
        if self.check_overlap(relays):
            return PENALTY_OVERLAP

        # --- 2. Symulacja Sieci ---
        total_energy_J = 0.0
        total_delay_s = 0.0
        min_link_margin_dB = 100.0 
        relay_usage = [0] * self.n_relays 
        
        for sensor in self.sensors:
            s_pos = np.array(sensor['pos'])
            s_zone, _ = BodyModel.get_zone_info(s_pos[0], s_pos[1])
            
            # Parametry Direct
            dist_dir = WBANPhysics.calculate_distance_m(s_pos, np.array(HUB_POS))
            pl_dir = WBANPhysics.calculate_path_loss_dB(dist_dir, s_zone[1] if s_zone else 'General')
            margin_dir = max(0, 96.0 - pl_dir)
            
            e_direct = WBANPhysics.calculate_energy_consumption(s_pos, np.array(HUB_POS), location_type=s_zone[1] if s_zone else 'General')
            d_direct = (1500 / 1_000_000)
            
            chosen_energy = e_direct
            chosen_delay = d_direct
            chosen_margin = margin_dir
            chosen_relay_idx = -1 

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

                e_relay_total = e_hop1 + e_hop2
                if e_relay_total < chosen_energy:
                    chosen_energy = e_relay_total
                    chosen_delay = d_hop1 + d_hop2 + 0.005
                    chosen_margin = min(margin_h1, margin_h2)
                    chosen_relay_idx = idx

            total_energy_J += chosen_energy * sensor['data_rate']
            total_delay_s += chosen_delay
            if chosen_margin < min_link_margin_dB:
                min_link_margin_dB = chosen_margin
            
            if chosen_relay_idx != -1:
                relay_usage[chosen_relay_idx] += 1

        f_energy = total_energy_J / NORM_FACTORS['energy']
        f_delay = total_delay_s / NORM_FACTORS['delay']
        f_quality = (100.0 - min_link_margin_dB) / NORM_FACTORS['quality']
        
        f_load = 0.0
        if sum(relay_usage) > 0:
            f_load = np.std(relay_usage) / NORM_FACTORS['load']

        fitness = (WEIGHTS['energy'] * f_energy +
                   WEIGHTS['delay']  * f_delay +
                   WEIGHTS['quality'] * f_quality +
                   WEIGHTS['load']   * f_load)
                   
        return fitness

    def get_metrics_details(self, solution_vector):
        """
        Zwraca słownik z fizycznymi wartościami metryk dla danego rozwiązania.
        """
        relays = self.decode_solution(solution_vector)
        
        # Sprawdź kary
        if not all(BodyModel.is_valid_position(r[0], r[1]) for r in relays) or self.check_overlap(relays):
            return {'Energy': float('nan'), 'Delay': float('nan'), 'Quality': 0.0}

        total_energy_J = 0.0
        total_delay_s = 0.0
        min_link_margin_dB = 100.0
        
        for sensor in self.sensors:
            s_pos = np.array(sensor['pos'])
            s_zone, _ = BodyModel.get_zone_info(s_pos[0], s_pos[1])
            
            # Direct
            dist_dir = WBANPhysics.calculate_distance_m(s_pos, np.array(HUB_POS))
            pl_dir = WBANPhysics.calculate_path_loss_dB(dist_dir, s_zone[1] if s_zone else 'General')
            margin_dir = max(0, 96.0 - pl_dir)
            e_direct = WBANPhysics.calculate_energy_consumption(s_pos, np.array(HUB_POS), location_type=s_zone[1] if s_zone else 'General')
            d_direct = (1500 / 1_000_000)
            
            chosen_energy = e_direct
            chosen_delay = d_direct
            chosen_margin = margin_dir
            
            for idx, r_pos in enumerate(relays):
                r_zone, r_type = BodyModel.get_zone_info(r_pos[0], r_pos[1])
                
                # Hop 1 & 2
                dist_h1 = WBANPhysics.calculate_distance_m(s_pos, r_pos)
                pl_h1 = WBANPhysics.calculate_path_loss_dB(dist_h1, s_zone[1] if s_zone else 'General')
                margin_h1 = max(0, 96.0 - pl_h1)
                e_hop1 = WBANPhysics.calculate_energy_consumption(s_pos, r_pos, location_type=s_zone[1] if s_zone else 'General')
                
                dist_h2 = WBANPhysics.calculate_distance_m(r_pos, np.array(HUB_POS))
                pl_h2 = WBANPhysics.calculate_path_loss_dB(dist_h2, r_type if r_zone else 'General')
                margin_h2 = max(0, 96.0 - pl_h2)
                e_hop2 = WBANPhysics.calculate_energy_consumption(r_pos, np.array(HUB_POS), location_type=r_type if r_zone else 'General')
                
                e_total = e_hop1 + e_hop2
                
                if e_total < chosen_energy:
                    chosen_energy = e_total
                    chosen_delay = (1500 / 1_000_000) * 2 + 0.005
                    chosen_margin = min(margin_h1, margin_h2)

            total_energy_J += chosen_energy * sensor['data_rate']
            total_delay_s += chosen_delay
            if chosen_margin < min_link_margin_dB:
                min_link_margin_dB = chosen_margin
                
        return {
            'Energy': total_energy_J,
            'Delay': total_delay_s,
            'Quality': min_link_margin_dB
        }
    
    def get_routing_details(self, solution_vector):
        """Metoda pomocnicza do wizualizacji"""
        relays = self.decode_solution(solution_vector)
        paths = []
        for sensor in self.sensors:
            s_pos = np.array(sensor['pos'])
            s_zone, _ = BodyModel.get_zone_info(s_pos[0], s_pos[1])
            e_direct = WBANPhysics.calculate_energy_consumption(s_pos, np.array(HUB_POS), location_type=s_zone[1] if s_zone else 'General')
            best_energy = e_direct
            chosen_path = {'from': s_pos, 'to': np.array(HUB_POS), 'type': 'Direct'}
            for idx, r_pos in enumerate(relays):
                e_hop1 = WBANPhysics.calculate_energy_consumption(s_pos, r_pos, location_type=s_zone[1] if s_zone else 'General')
                r_zone, r_type = BodyModel.get_zone_info(r_pos[0], r_pos[1])
                e_hop2 = WBANPhysics.calculate_energy_consumption(r_pos, np.array(HUB_POS), location_type=r_type if r_zone else 'General')
                if (e_hop1 + e_hop2) < best_energy:
                    best_energy = e_hop1 + e_hop2
                    chosen_path = [
                        {'from': s_pos, 'to': r_pos, 'type': 'Relay'},
                        {'from': r_pos, 'to': np.array(HUB_POS), 'type': 'Relay'}
                    ]
            if isinstance(chosen_path, list):
                paths.extend(chosen_path)
            else:
                paths.append(chosen_path)
        return paths