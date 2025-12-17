import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.body_model import ALLOWED_ZONES, LANDMARKS
from src.fitness import FIXED_SENSORS, HUB_POS
from src.physics import WBANPhysics

# --- KONFIGURACJA STYLU IEEE / NAUKOWEGO ---
plt.rcParams.update({
    'font.family': 'serif',          # Czcionka szeryfowa (jak w LaTeX/Word)
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,               # Wysoka rozdzielczość do druku
    'lines.linewidth': 1.5,
    'lines.markersize': 8
})

def plot_convergence(history, algorithm_name="PSO"):
    """
    Rysuje wykres zbieżności w stylu naukowym.
    """
    global_bests = history.list_global_best_fit
    epochs = range(1, len(global_bests) + 1)

    plt.figure(figsize=(8, 5)) # Proporcje 8x5 są dobre do prac A4
    
    # Styl linii: ciągła z markerami co 5 punktów (żeby nie zamazać wykresu)
    plt.plot(epochs, global_bests, linestyle='-', color='black', linewidth=1.5, label='Best Fitness Value')
    plt.scatter(epochs[::5], global_bests[::5], marker='o', color='black', s=30) # Markery co 5 kroków

    plt.title(f'Convergence Curve - {algorithm_name}')
    plt.xlabel('Iteration (Epoch)')
    plt.ylabel('Cost Function Value (Fitness)')
    plt.grid(True, linestyle='--', alpha=0.5) # Delikatna siatka
    plt.legend(frameon=True, loc='upper right')
    
    plt.tight_layout()
    filename = f"convergence_{algorithm_name}.png"
    plt.savefig(filename, dpi=300)
    print(f"[INFO] Wykres zbieżności zapisano jako: {filename}")
    plt.close()

def plot_body_simulation(relays_pos, active_paths=None, title="WBAN Topology Optimization"):
    """
    Rysuje mapę ciała i topologię sieci w stylu naukowym.
    """
    fig, ax = plt.subplots(figsize=(7, 10)) # Format pionowy
    
    # 1. Rysuj Strefy (Tło) - Używamy odcieni szarości dla czytelności w druku
    # 'type': color
    zone_colors = {'LOS': '#f0f0f0', 'NLOS': '#e0e0e0', 'Torso': '#d0d0d0'}
    
    # Dodajemy legendę stref ręcznie, żeby uniknąć duplikatów
    added_zone_labels = set()

    for name, data in ALLOWED_ZONES.items():
        b = data['bounds']
        width = b[1] - b[0]
        height = b[3] - b[2]
        
        z_type = data['type']
        label = f"Zone: {z_type}"
        
        # Logika unikania duplikatów w legendzie
        if label in added_zone_labels:
            lbl_arg = "_nolegend_"
        else:
            lbl_arg = label
            added_zone_labels.add(label)

        rect = patches.Rectangle((b[0], b[2]), width, height, 
                                 linewidth=0.5, edgecolor='gray', 
                                 facecolor=zone_colors.get(z_type, 'white'), 
                                 alpha=0.6, label=lbl_arg)
        ax.add_patch(rect)
        
        # Nazwa strefy na wykresie (mała czcionka)
        ax.text(b[0] + width/2, b[2] + height/2, name.replace("_", "\n"), 
                ha='center', va='center', fontsize=6, color='#555555')

    # 2. Rysuj Hub (Sink)
    ax.scatter(HUB_POS[0], HUB_POS[1], c='black', s=150, marker='P', label='HUB (Sink)', zorder=10)
    
    # 3. Rysuj Sensory
    added_sensor_label = False
    for s in FIXED_SENSORS:
        pos = s['pos']
        lbl = 'Medical Sensor' if not added_sensor_label else "_nolegend_"
        ax.scatter(pos[0], pos[1], c='white', s=80, marker='o', edgecolors='black', linewidth=1.5, zorder=9, label=lbl)
        added_sensor_label = True
        # Podpis sensora
        ax.text(pos[0]+3, pos[1], s['name'].replace("_", " "), fontsize=8, color='black', verticalalignment='center')

    # 4. Rysuj Relaye (Cluster Heads)
    for i, pos in enumerate(relays_pos):
        lbl = 'Optimized Relay (CH)' if i == 0 else "_nolegend_"
        ax.scatter(pos[0], pos[1], c='black', s=120, marker='^', zorder=11, label=lbl)
        ax.text(pos[0]+3, pos[1], f"CH-{i+1}", fontsize=9, fontweight='bold', color='black', verticalalignment='center')

    # 5. Rysuj Połączenia (Active Paths)
    if active_paths:
        added_direct = False
        added_relay = False
        
        for path in active_paths:
            p_from = path['from']
            p_to = path['to']
            p_type = path['type']
            
            if p_type == 'Direct':
                style = ':' # Kropkowana dla Direct
                color = '#444444'
                width = 1.0
                lbl = 'Direct Path' if not added_direct else "_nolegend_"
                if not added_direct: added_direct = True
            else:
                style = '-' # Ciągła dla Relay
                color = 'black'
                width = 1.5
                lbl = 'Relayed Path' if not added_relay else "_nolegend_"
                if not added_relay: added_relay = True
            
            ax.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]], 
                    linestyle=style, color=color, linewidth=width, alpha=0.7, zorder=5, label=lbl)

    # Konfiguracja osi
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 180)
    ax.set_aspect('equal')
    ax.set_xlabel("Body Width X [cm]")
    ax.set_ylabel("Body Height Y [cm]")
    ax.set_title(title, pad=15)
    
    # Legenda na dole, pozioma
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)
    
    # Grid
    ax.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "simulation_result_IEEE.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Mapa topologii zapisana jako: {output_path}")
    plt.close()