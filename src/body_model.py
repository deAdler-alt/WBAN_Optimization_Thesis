import numpy as np
import random

# ==================================================================================
# MAPA CIAŁA (UNFOLDED MODEL)
# Oparta na danych z Tabeli 2 raportu [WBAN Optimization Literature Search, 2025]
# Przeliczono mm -> cm dla uproszczenia
# ==================================================================================

# Punkty referencyjne (Środki stref)
# Zakładamy model 2D, gdzie X rozdziela strefy (np. X < 50 to przód, X > 50 to tył)
LANDMARKS = {
    'CHEST':     (36.0, 35.0), # Klatka piersiowa (Przód)
    'NAVEL':     (40.0, 54.0), # Pępek (Sink / Hub)
    'WRIST_L':   (15.0, 54.0), # Lewy nadgarstek (odunięty od tułowia na mapie)
    'ANKLE_L':   (21.0, 109.0),# Lewa kostka
    'BACK':      (70.0, 35.0)  # Plecy (Symulowane przesunięcie X o +34cm względem klatki)
}

# Definicja DOZWOLONYCH STREF (Constraints)
# Algorytm może wybrać punkt (x,y) tylko jeśli wpada w jeden z tych prostokątów.
# Format: [x_min, x_max, y_min, y_max, 'Physics_Type']
ALLOWED_ZONES = {
    'TORSO_FRONT': {
        'bounds': [25.0, 50.0, 20.0, 60.0], # Obszar klatki i brzucha
        'type': 'Torso' # Duże tłumienie
    },
    'BACK_ZONE': {
        'bounds': [60.0, 85.0, 20.0, 60.0], # Obszar pleców
        'type': 'LOS'   # Małe tłumienie (Path Loss Exponent n=2.18)
    },
    'ARM_LEFT': {
        'bounds': [5.0, 20.0, 40.0, 70.0],  # Obszar ręki
        'type': 'NLOS'  # Średnie tłumienie
    },
    'LEG_LEFT': {
        'bounds': [10.0, 30.0, 80.0, 120.0], # Obszar nogi
        'type': 'NLOS'
    }
}

class BodyModel:
    """
    Reprezentuje model ciała i ograniczenia geometryczne.
    """
    
    @staticmethod
    def get_zone_info(x, y):
        """
        Sprawdza, w jakiej strefie znajduje się punkt (x,y).
        Zwraca: (nazwa_strefy, typ_fizyczny) lub (None, None) jeśli poza ciałem.
        """
        for zone_name, data in ALLOWED_ZONES.items():
            b = data['bounds']
            # Sprawdź czy x, y mieści się w prostokącie [xmin, xmax, ymin, ymax]
            if b[0] <= x <= b[1] and b[2] <= y <= b[3]:
                return zone_name, data['type']
        
        return None, None # Punkt poza dozwolonym obszarem (np. w powietrzu)

    @staticmethod
    def is_valid_position(x, y):
        """Czy punkt jest poprawny?"""
        zone, _ = BodyModel.get_zone_info(x, y)
        return zone is not None

    @staticmethod
    def get_random_valid_position():
        """
        Losuje poprawny punkt na ciele (do inicjalizacji populacji GA/PSO).
        """
        # 1. Wylosuj strefę (np. Ręka, Noga, Plecy)
        zone_name = random.choice(list(ALLOWED_ZONES.keys()))
        data = ALLOWED_ZONES[zone_name]
        b = data['bounds']
        
        # 2. Wylosuj punkt wewnątrz tej strefy
        x = random.uniform(b[0], b[1])
        y = random.uniform(b[2], b[3])
        return np.array([x, y])

    @staticmethod
    def get_hub_position():
        """Zwraca domyślną pozycję Huba (Pępek/Pas)"""
        # Możemy przyjąć stałą pozycję z raportu
        return np.array(LANDMARKS['NAVEL'])

# --- TEST WERYFIKACYJNY ---
if __name__ == "__main__":
    print("--- Body Model Test ---")
    
    # 1. Test Huba
    hub = BodyModel.get_hub_position()
    zone, ptype = BodyModel.get_zone_info(hub[0], hub[1])
    print(f"Hub Position: {hub} -> Strefa: {zone}, Typ: {ptype}")
    
    # 2. Test losowania punktów
    print("\nLosowanie 5 punktów sensorów:")
    for i in range(5):
        pos = BodyModel.get_random_valid_position()
        z, t = BodyModel.get_zone_info(pos[0], pos[1])
        print(f"  Sensor {i+1}: ({pos[0]:.1f}, {pos[1]:.1f}) -> {z} [{t}]")
        
    # 3. Test poprawności fizycznej (integracja logiczna)
    # Sprawdzamy czy punkt na plecach (X=70, Y=35) daje typ 'LOS'
    test_back = (70.0, 35.0)
    _, t_back = BodyModel.get_zone_info(*test_back)
    
    if t_back == 'LOS':
        print("\n>> SUKCES: Mapa poprawnie identyfikuje plecy (LOS).")
    else:
        print(f"\n>> BŁĄD: Plecy powinny być LOS, a są {t_back}")