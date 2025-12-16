import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.body_model import ALLOWED_ZONES, LANDMARKS
from src.fitness import FIXED_SENSORS, HUB_POS

def plot_body_simulation(relays_pos=None, title="WBAN Simulation Result"):
    """
    Rysuje mapę ciała (strefy), stałe sensory i (opcjonalnie) pozycje Relayów.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # 1. Rysuj Strefy (Allowed Zones)
    colors = {'LOS': 'lightgreen', 'NLOS': 'lightyellow', 'Torso': 'mistyrose'}
    
    for name, data in ALLOWED_ZONES.items():
        b = data['bounds'] # [xmin, xmax, ymin, ymax]
        width = b[1] - b[0]
        height = b[3] - b[2]
        rect = patches.Rectangle((b[0], b[2]), width, height, 
                                 linewidth=1, edgecolor='gray', facecolor=colors.get(data['type'], 'lightgray'), 
                                 alpha=0.5, label=f"{data['type']}" if f"{data['type']}" not in [l.get_label() for l in ax.patches] else "")
        ax.add_patch(rect)
        # Podpis strefy
        ax.text(b[0] + width/2, b[2] + height/2, name, 
                ha='center', va='center', fontsize=8, color='black', alpha=0.7)

    # 2. Rysuj Hub
    ax.scatter(HUB_POS[0], HUB_POS[1], c='red', s=150, marker='X', label='HUB (Sink)', zorder=10)
    
    # 3. Rysuj Stałe Sensory
    for s in FIXED_SENSORS:
        pos = s['pos']
        ax.scatter(pos[0], pos[1], c='blue', s=80, marker='o', zorder=9)
        ax.text(pos[0]+2, pos[1], s['name'], fontsize=9, color='blue')

    # 4. Rysuj Relaye (jeśli podano)
    if relays_pos:
        for i, pos in enumerate(relays_pos):
            ax.scatter(pos[0], pos[1], c='purple', s=120, marker='^', label='Optimized Relay' if i==0 else "", zorder=11)
            ax.text(pos[0]+2, pos[1], f"Relay {i+1}", fontsize=9, color='purple', fontweight='bold')
            
            # Opcjonalnie: Linia do Huba (wizualizacja połączenia)
            ax.plot([pos[0], HUB_POS[0]], [pos[1], HUB_POS[1]], 'k--', alpha=0.3, linewidth=0.8)

    # Legenda i opisy
    # Usuń duplikaty z legendy
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)
    ax.set_xlabel("X [cm] (Unfolded Body Width)")
    ax.set_ylabel("Y [cm] (Height)")
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Zapisz do pliku zamiast wyświetlać (bezpieczniej w terminalu)
    output_path = "simulation_result.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[INFO] Wykres zapisano jako: {output_path}")
    # plt.show() # Odkomentuj jeśli masz środowisko graficzne

if __name__ == "__main__":
    # Test: Rysuje mapę z losowymi relayami
    test_relays = [[20, 100], [70, 40]]
    plot_body_simulation(test_relays, title="Test Visualization")