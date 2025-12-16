import numpy as np

# ==================================================================================
# KONFIGURACJA PARAMETRÓW FIZYCZNYCH (WBAN - IEEE 802.15.6 + nRF52840)
# ==================================================================================

# 1. Parametry Radia (wzorowane na Nordic nRF52840 - typowy chip IoT/WBAN)
VOLTAGE = 3.0           # V
BIT_RATE = 1_000_000    # 1 Mbps
RX_SENSITIVITY = -96.0  # dBm (Bardzo czułe radio)
SYSTEM_MARGIN = 10.0    # dB (Margines na zaniki sygnału)

# Limity mocy nadawania (dBm)
TX_POWER_MIN = -40.0    # Tryb super-low power
TX_POWER_MAX = 4.0      # Maksymalna moc

# 2. Parametry modelu propagacji IEEE 802.15.6 (CM3)
# [Chavez et al. 2013]
IEEE_802_15_6_PARAMS = {
    'LOS':   {'n': 2.18, 'sigma': 5.6},  # Plecy (Dobra propagacja)
    'NLOS':  {'n': 3.35, 'sigma': 4.1},  # Kończyny (Średnia propagacja)
    'Torso': {'n': 3.23, 'sigma': 6.1},  # Klatka/Tułów (Duże tłumienie)
    'General': {'n': 3.11, 'sigma': 5.9}
}

class WBANPhysics:
    
    @staticmethod
    def get_path_loss_params(location_type):
        return IEEE_802_15_6_PARAMS.get(location_type, IEEE_802_15_6_PARAMS['General'])

    @staticmethod
    def calculate_distance_m(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        dist_cm = np.linalg.norm(p1 - p2)
        # Zabezpieczenie: minimalny dystans 1 cm (dla fizyki anteny)
        return max(dist_cm / 100.0, 0.01)

    @staticmethod
    def calculate_path_loss_dB(distance_m, location_type='General'):
        """
        Log-Normal Shadowing Path Loss
        """
        d0 = 0.1  # 10 cm reference
        PL_d0 = 35.0 # dB @ 2.4GHz
        
        params = WBANPhysics.get_path_loss_params(location_type)
        n = params['n']
        
        if distance_m <= d0:
            return PL_d0
            
        # W optymalizacji używamy wartości średniej (bez losowego shadowingu),
        # żeby algorytm był deterministyczny (łatwiejszy do debugowania).
        pl = PL_d0 + 10 * n * np.log10(distance_m / d0)
        return pl

    @staticmethod
    def calculate_energy_consumption(p1, p2, location_type='General', packet_size_bits=1500):
        """
        Zwraca energię [J] dla jednego pakietu z adaptacją mocy (Tx Power Control).
        """
        dist = WBANPhysics.calculate_distance_m(p1, p2)
        
        # 1. Tłumienie kanału
        pl_dB = WBANPhysics.calculate_path_loss_dB(dist, location_type)
        
        # 2. Wymagana moc nadawania
        required_tx_dBm = RX_SENSITIVITY + pl_dB + SYSTEM_MARGIN
        
        # 3. Dopasowanie do możliwości sprzętu (-40 do +4 dBm)
        tx_power_dBm = np.clip(required_tx_dBm, TX_POWER_MIN, TX_POWER_MAX)
        
        # 4. Model prądu (mA) w funkcji mocy (dBm)
        # Aproksymacja liniowa dla nRF52840 w zakresie -40..+4 dBm:
        # -40 dBm -> ~3.0 mA (min base current)
        #   0 dBm -> ~5.0 mA
        #  +4 dBm -> ~7.5 mA
        # Wzór: I(mA) = 3.0 + 0.1 * (Tx_dBm + 40)
        current_mA = 3.0 + 0.1 * (tx_power_dBm + 40.0)
        current_A = current_mA / 1000.0
        
        # 5. Czas lotu pakietu
        time_s = packet_size_bits / BIT_RATE
        
        # 6. Energia całkowita (Tx + stały koszt Rx po stronie huba pomijamy lub dodajemy stałą)
        # Tutaj liczymy koszt energetyczny SAMEGO SENSORA (nadawcy)
        energy_J = VOLTAGE * current_A * time_s
        
        return energy_J

# --- TEST WERYFIKACYJNY ---
if __name__ == "__main__":
    print("--- WBAN Physics Test (v2 - High Sensitivity) ---")
    
    # Scenariusz: Czujnik na kostce -> Hub na pasie/klatce
    # Dystans: 120 cm (1.2 m)
    p1 = (0, 0)
    p2 = (0, 120) 
    
    # 1. Transmisja po plecach (np. wzdłuż kręgosłupa - 'LOS')
    e_back = WBANPhysics.calculate_energy_consumption(p1, p2, 'LOS')
    
    # 2. Transmisja przez klatkę/brzuch (tkanki tłumiące - 'Torso')
    e_torso = WBANPhysics.calculate_energy_consumption(p1, p2, 'Torso')
    
    print(f"Dystans: 120 cm")
    print(f"Energia [Plecy, n=2.18]:  {e_back:.3e} J")
    print(f"Energia [Klatka, n=3.23]: {e_torso:.3e} J")
    
    diff = ((e_torso - e_back) / e_torso) * 100
    print(f"ZYSK z umieszczenia na plecach: {diff:.2f}%")
    
    if diff > 1.0:
        print(">> SUKCES: Fizyka działa! Model widzi różnicę.")
    else:
        print(">> OSTRZEŻENIE: Nadal mała różnica (sprawdź dystans lub parametry).")