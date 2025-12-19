import numpy as np

# ==============================================================================
# PARAMETRY Z FIZYKI (Kopia z src/physics.py dla pewności testu)
# ==============================================================================
VOLTAGE = 3.0           # V
BIT_RATE = 1_000_000    # 1 Mbps
RX_SENSITIVITY = -96.0  # dBm
SYSTEM_MARGIN = 10.0    # dB
TX_POWER_MIN = -40.0    # dBm
TX_POWER_MAX = 4.0      # dBm
PACKET_SIZE = 1500      # bity

# IEEE 802.15.6 Path Loss Exponents
n_LOS = 2.18   # Line of Sight (Plecy)
n_NLOS = 3.35  # Non-Line of Sight (Kończyny)
n_Torso = 3.23 # Tułów (Klatka)

PL_d0 = 35.0   # dB @ 10cm
d0 = 0.1       # m

def calculate_cost(distance_m, n_exponent, label):
    """Oblicza Path Loss, Moc Tx i Energię dla zadanego dystansu i n."""
    
    # 1. Path Loss Model
    if distance_m <= d0:
        pl_dB = PL_d0
    else:
        pl_dB = PL_d0 + 10 * n_exponent * np.log10(distance_m / d0)
    
    # 2. Wymagana moc nadawania
    req_tx_dBm = RX_SENSITIVITY + pl_dB + SYSTEM_MARGIN
    
    # 3. Clip do możliwości sprzętu
    # To jest kluczowe! Jeśli req > 4.0, to mamy problem (ale liczymy max energię)
    final_tx_dBm = np.clip(req_tx_dBm, TX_POWER_MIN, TX_POWER_MAX)
    
    # Czy link jest możliwy? (Czy wymagana moc nie przekracza max radia?)
    is_connected = req_tx_dBm <= TX_POWER_MAX
    
    # 4. Prąd (Model liniowy nRF52)
    # I(mA) ≈ 3.0 + 0.1 * (P_dBm + 40)
    current_mA = 3.0 + 0.1 * (final_tx_dBm + 40.0)
    current_A = current_mA / 1000.0
    
    # 5. Czas
    time_s = PACKET_SIZE / BIT_RATE
    
    # 6. Energia
    energy_J = VOLTAGE * current_A * time_s
    
    return {
        "Scenariusz": label,
        "Dystans": f"{distance_m*100:.0f} cm",
        "n": n_exponent,
        "PathLoss": f"{pl_dB:.2f} dB",
        "Req_Tx": f"{req_tx_dBm:.2f} dBm",
        "Final_Tx": f"{final_tx_dBm:.2f} dBm",
        "Connected": "TAK" if is_connected else "NIE (Zasięg!)",
        "Energy": f"{energy_J:.3e} J"
    }

# ==============================================================================
# URUCHOMIENIE TESTU
# ==============================================================================
if __name__ == "__main__":
    print(f"{'Scenariusz':<15} | {'Dist':<8} | {'n':<4} | {'PathLoss':<10} | {'Req Tx':<10} | {'Status':<12} | {'ENERGY (Cel)'}")
    print("-" * 90)
    
    test_cases = [
        (0.1, n_LOS, "Blisko (Ref)"),
        (0.5, n_LOS, "Plecy (LOS)"),
        (0.5, n_Torso, "Klatka (Torso)"),
        (1.0, n_LOS, "Plecy (LOS)"),
        (1.0, n_NLOS, "Noga (NLOS)"),
        (1.5, n_NLOS, "Skraj (NLOS)")
    ]
    
    for dist, n, lab in test_cases:
        res = calculate_cost(dist, n, lab)
        print(f"{res['Scenariusz']:<15} | {res['Dystans']:<8} | {res['n']:<4} | {res['PathLoss']:<10} | {res['Req_Tx']:<10} | {res['Connected']:<12} | {res['Energy']}")

    print("-" * 90)
    print("WNIOSKI DO WERYFIKACJI:")
    print("1. Czy Energia rośnie wraz z dystansem?")
    print("2. Czy dla tego samego dystansu (np. 50cm) Klatka (Torso) zużywa więcej niż Plecy (LOS)?")
    print("3. Czy przy 150cm (Skraj) łącze jest zrywane (Status: NIE)?")
