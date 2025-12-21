import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==============================================================================
# 1. KONFIGURACJA STYLU (POLSKI, NAUKOWY)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'figure.dpi': 300
})

OUTPUT_DIR = "FINAL_CHARTS_REAL_DATA"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ALGOS = ['GA', 'PSO', 'GWO']
COLORS = {'GA': '#d62728', 'PSO': '#1f77b4', 'GWO': '#2ca02c'}
MARKERS = {'GA': 's', 'PSO': '^', 'GWO': 'o'}

print(">>> Wczytywanie danych z plików CSV...")

# ==============================================================================
# 2. WCZYTYWANIE I NAPRAWA DANYCH
# ==============================================================================
try:
    df = pd.read_csv('WBAN_Experiment_Results.csv')
    
    # Przeliczenie opóźnienia na milisekundy (zabezpieczenie przed zerami)
    # Jeśli wynik jest np. 1e-15, traktujemy to jako 0.0
    df['Delay_ms'] = df['Avg_Delay_s'] * 1000 
    
    # Filtrowanie
    df = df[df['Algorithm'].isin(ALGOS)]
    
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku WBAN_Experiment_Results.csv!")
    print("Upewnij się, że plik jest w tym samym folderze co skrypt.")
    exit()

try:
    df_sens = pd.read_csv('WBAN_Sensitivity_Results.csv')
    df_sens = df_sens[df_sens['Algorithm'].isin(ALGOS)]
    df_sens['Success_Pct'] = df_sens['Is_Success'] * 100
except FileNotFoundError:
    print("OSTRZEŻENIE: Nie znaleziono WBAN_Sensitivity_Results.csv. Pominę wykres sukcesu.")
    df_sens = pd.DataFrame()

# ==============================================================================
# 3. FUNKCJA ZAPISU I GENEROWANIE WYKRESÓW
# ==============================================================================
def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"Wygenerowano: {path}")
    plt.close()

# --- WYKRES 1: ENERGIA ---
plt.figure(figsize=(10, 6))
# POPRAWKA: capsize przeniesione do err_kws
sns.lineplot(data=df, x='Scenario_Sensors', y='Energy_Total_J', hue='Algorithm', style='Algorithm',
             palette=COLORS, markers=MARKERS, err_style="bars", errorbar=("sd", 1), 
             err_kws={'capsize': 5}) 
plt.xlabel("Liczba Sensorów")
plt.ylabel("Energia Całkowita [J]")
plt.title("Rys. 5.1. Trend zużycia energii (Średnia ± Odchylenie Std.)")
plt.grid(True, alpha=0.3)
save_plot("Fig1_Energy_Trend.png")

# --- WYKRES 2: OPÓŹNIENIE ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Scenario_Sensors', y='Delay_ms', hue='Algorithm', style='Algorithm',
             palette=COLORS, markers=MARKERS, err_style="bars", errorbar=("sd", 1),
             err_kws={'capsize': 5})
plt.xlabel("Liczba Sensorów")
plt.ylabel("Opóźnienie [ms]")
plt.title("Rys. 5.6. Średnie opóźnienie transmisji")
plt.grid(True, alpha=0.3)
save_plot("Fig2_Delay_Trend.png")

# --- WYKRES 3: LINK MARGIN ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Scenario_Sensors', y='Min_Link_Margin_dB', hue='Algorithm', style='Algorithm',
             palette=COLORS, markers=MARKERS, err_style="bars", errorbar=("sd", 1),
             err_kws={'capsize': 5})
plt.xlabel("Liczba Sensorów")
plt.ylabel("Link Margin [dB]")
plt.title("Rys. 5.3. Margines łącza radiowego (Link Margin)")
plt.grid(True, alpha=0.3)
save_plot("Fig4_Reliability.png")

# --- WYKRES 4: CZAS OBLICZEŃ ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Scenario_Sensors', y='Execution_Time_s', hue='Algorithm', style='Algorithm',
             palette=COLORS, markers=MARKERS, err_style="bars", errorbar=("sd", 1),
             err_kws={'capsize': 5})
plt.xlabel("Liczba Sensorów")
plt.ylabel("Czas Obliczeń [s]")
plt.title("Rys. 5.4. Koszt obliczeniowy (Czas zbieżności)")
plt.grid(True, alpha=0.3)
save_plot("Fig3_Time_Cost.png")

# --- WYKRES 5: SUCCESS RATE (Słupkowy) ---
if not df_sens.empty:
    sens_agg = df_sens.groupby(['Config_Pack', 'Algorithm'])['Success_Pct'].mean().reset_index()
    
    # Mapowanie nazw
    pack_order = ['A_Eco', 'B_Standard', 'C_High']
    labels_map = {'A_Eco': 'Eco', 'B_Standard': 'Standard', 'C_High': 'High'}
    sens_agg['Config_Label'] = sens_agg['Config_Pack'].map(labels_map)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=sens_agg, x='Config_Label', y='Success_Pct', hue='Algorithm', 
                palette=COLORS, order=['Eco', 'Standard', 'High'], edgecolor='black', errorbar=None)
    plt.xlabel("Dostępne Zasoby (Tryb)")
    plt.ylabel("Skuteczność [%]")
    plt.title("Rys. 5.7. Skuteczność algorytmów (Success Rate)")
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    save_plot("Fig6_Sensitivity.png")

# --- WYKRES 6: ENERGIA SŁUPKOWY (PORÓWNANIE) ---
subset = df[df['Scenario_Sensors'].isin([6, 12, 20])]
plt.figure(figsize=(9, 6))
sns.barplot(data=subset, x='Scenario_Sensors', y='Energy_Total_J', hue='Algorithm',
            palette=COLORS, edgecolor='black', errorbar=None)
plt.xlabel("Rozmiar Sieci (Liczba Sensorów)")
plt.ylabel("Średnie Zużycie Energii [J]")
plt.title("Rys. 5.2. Porównanie Efektywności Energetycznej")
plt.grid(axis='y', alpha=0.3)
save_plot("Fig5_Energy_Comparison.png")

print("\n>>> SUKCES! Wszystkie wykresy wygenerowane poprawnie.")