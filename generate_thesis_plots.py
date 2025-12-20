import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# 1. KONFIGURACJA STYLU (CZYTELNOŚĆ + JITTER)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'figure.dpi': 300,
    'figure.constrained_layout.use': True
})

OUTPUT_DIR = "FINAL_THESIS_CHARTS_V5"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SELECTED_ALGOS = ['GA', 'PSO', 'GWO']

# KONFIGURACJA WIZUALNA (JITTER + STYLE)
# offset: Przesunięcie punktów w lewo/prawo, żeby się nie nakładały
ALGO_OPTS = {
    'GA':  {'color': '#d62728', 'marker': 's', 'label': 'GA',  'ls': '-',  'offset': 0.0},   # Czerwony, Ciągła
    'PSO': {'color': '#1f77b4', 'marker': '^', 'label': 'PSO', 'ls': '--', 'offset': -0.2}, # Niebieski, Przerywana, Lewo
    'GWO': {'color': '#2ca02c', 'marker': 'o', 'label': 'GWO', 'ls': ':',  'offset': 0.2}   # Zielony, Kropkowana, Prawo
}

def save_current_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    print(f"[GENERATED] {path}")
    plt.close()

# ==============================================================================
# 2. PRZYGOTOWANIE DANYCH
# ==============================================================================
print(">>> Wczytywanie danych...")
try:
    df_exp = pd.read_csv('WBAN_Experiment_Results.csv')
    df_sens = pd.read_csv('WBAN_Sensitivity_Results.csv')
except FileNotFoundError:
    print("BŁĄD: Brakuje plików CSV.")
    exit()

df_exp = df_exp[df_exp['Algorithm'].isin(SELECTED_ALGOS)]
df_sens = df_sens[df_sens['Algorithm'].isin(SELECTED_ALGOS)]
df_valid = df_exp.dropna(subset=['Energy_Total_J'])

# ==============================================================================
# 3. WYKRESY LINIOWE Z PRZESUNIĘCIEM (JITTERED LINE CHARTS)
# ==============================================================================
def plot_jittered_line(metric, y_label, title, filename):
    plt.figure(figsize=(8, 5))
    data = df_valid.groupby(['Scenario_Sensors', 'Algorithm'])[metric].mean().reset_index()
    
    for algo in SELECTED_ALGOS:
        subset = data[data['Algorithm'] == algo]
        opts = ALGO_OPTS[algo]
        
        # JITTER: Przesuwamy X o małą wartość
        x_shifted = subset['Scenario_Sensors'] + opts['offset']
        
        plt.plot(x_shifted, subset[metric],
                 label=opts['label'], color=opts['color'],
                 marker=opts['marker'], linestyle=opts['ls'],
                 alpha=0.9) # Lekka przezroczystość pomaga widzieć przecięcia

    plt.xlabel('Liczba Sensorów')
    plt.ylabel(y_label)
    plt.title(title)
    
    # Naprawiamy etykiety osi X (żeby pokazywały równe liczby mimo przesunięcia)
    unique_sensors = sorted(df_valid['Scenario_Sensors'].unique())
    plt.xticks(unique_sensors)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_current_plot(filename)

plot_jittered_line('Energy_Total_J', 'Energia Całkowita [J]', 'Fig 1. Trend Energii (Skalowalność)', 'Fig1_Energy_Trend.png')
plot_jittered_line('Avg_Delay_s', 'Opóźnienie [s]', 'Fig 2. Opóźnienie Transmisji', 'Fig2_Delay_Trend.png')
plot_jittered_line('Execution_Time_s', 'Czas Obliczeń [s]', 'Fig 3. Koszt Obliczeniowy', 'Fig3_Time_Cost.png')
plot_jittered_line('Min_Link_Margin_dB', 'Link Margin [dB]', 'Fig 4. Niezawodność Łącza', 'Fig4_Reliability.png')

# ==============================================================================
# 4. NOWY FIG 5: BEZPOŚREDNIE PORÓWNANIE ENERGII (GROUPED BAR)
# ==============================================================================
# Wybieramy 3 reprezentatywne scenariusze: Mały (6), Średni (12), Duży (20)
scenarios = [6, 12, 20]
x = np.arange(len(scenarios))
width = 0.25

plt.figure(figsize=(9, 6))

for i, algo in enumerate(SELECTED_ALGOS):
    means = []
    for s in scenarios:
        val = df_valid[(df_valid['Scenario_Sensors'] == s) & (df_valid['Algorithm'] == algo)]['Energy_Total_J'].mean()
        means.append(val if not np.isnan(val) else 0)
    
    offset = (i - 1) * width
    bars = plt.bar(x + offset, means, width, 
                   label=ALGO_OPTS[algo]['label'], 
                   color=ALGO_OPTS[algo]['color'], 
                   edgecolor='black', alpha=0.9)
    
    # Dodajemy wartości na słupkach tylko dla 20 sensorów (żeby nie zaciemniać)
    if i == 1: # Dla środkowego słupka (PSO) dodaj wartość jako odniesienie
       pass 

plt.xlabel('Rozmiar Sieci (Liczba Sensorów)')
plt.ylabel('Średnie Zużycie Energii [J]')
plt.title('Fig 5. Porównanie Efektywności Energetycznej')
plt.xticks(x, [f'{s} Sensorów' for s in scenarios])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Dodajemy komentarz na wykresie
plt.text(1, max(means)*0.9, "Wszystkie algorytmy osiągają\nto samo optimum fizyczne", 
         ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

save_current_plot('Fig5_Energy_Comparison.png')

# ==============================================================================
# 5. FIG 6: ANALIZA WRAŻLIWOŚCI (BEZ ZMIAN)
# ==============================================================================
success_data = df_sens.groupby(['Config_Pack', 'Algorithm'])['Is_Success'].mean().reset_index()
success_data['Success_Pct'] = success_data['Is_Success'] * 100
packs = ['A_Eco', 'B_Standard', 'C_High']
x = np.arange(len(packs))
width = 0.25

plt.figure(figsize=(9, 6))
for i, algo in enumerate(SELECTED_ALGOS):
    vals = [success_data[(success_data['Algorithm']==algo) & (success_data['Config_Pack']==p)]['Success_Pct'].values[0] if not success_data[(success_data['Algorithm']==algo) & (success_data['Config_Pack']==p)].empty else 0 for p in packs]
    offset = (i - 1) * width
    plt.bar(x + offset, vals, width, label=ALGO_OPTS[algo]['label'], color=ALGO_OPTS[algo]['color'], edgecolor='black')

plt.xlabel('Zasoby (Paczki)')
plt.ylabel('Sukces [%]')
plt.title('Fig 6. Stabilność Algorytmów')
plt.xticks(x, ['Eco', 'Standard', 'High'])
plt.ylim(0, 110)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
save_current_plot('Fig6_Sensitivity.png')

print(f"\n[SUKCES] Nowe wykresy (Jitter + Grouped Energy) w folderze '{OUTPUT_DIR}'.")