import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.ticker as ticker

# ==============================================================================
# 1. KONFIGURACJA STYLU (NAUKOWY / IEEE)
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
    'figure.dpi': 300,
    'axes.formatter.useoffset': False # WYŁĄCZENIE NOTACJI OFFSETOWEJ (DLA OPÓŹNIEŃ)
})

OUTPUT_DIR = "FINAL_THESIS_PLOTS_V3"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ALGOS = ['GA', 'PSO', 'GWO']
COLORS = {'GA': '#d62728', 'PSO': '#1f77b4', 'GWO': '#2ca02c'}
MARKERS = {'GA': 's', 'PSO': '^', 'GWO': 'o'}

print(">>> Wczytywanie i czyszczenie danych...")

# ==============================================================================
# 2. WCZYTYWANIE DANYCH Z FILTROWANIEM BŁĘDÓW
# ==============================================================================
try:
    df = pd.read_csv('WBAN_Experiment_Results.csv')
    
    # 1. Konwersja opóźnienia na ms
    df['Delay_ms'] = df['Avg_Delay_s'] * 1000 
    
    # 2. FILTROWANIE NIEUDANYCH PRÓB (KLUCZOWE DLA GA!)
    # Usuwamy wiersze, gdzie Link Margin wynosi 0.0 (brak połączenia)
    # To naprawi "wielkie słupki błędów" wynikające z uśredniania zer
    df_clean = df[df['Min_Link_Margin_dB'] > 0.1].copy()
    
    # Wyświetlenie statystyk odrzuconych prób
    rejected = len(df) - len(df_clean)
    print(f"Odrzucono {rejected} prób (nieudane/zerwane połączenia), głównie dla GA.")
    
    # Filtrowanie algorytmów
    df_clean = df_clean[df_clean['Algorithm'].isin(ALGOS)]
    
except FileNotFoundError:
    print("BŁĄD: Brak pliku WBAN_Experiment_Results.csv")
    exit()

# Dane do Success Rate
try:
    df_sens = pd.read_csv('WBAN_Sensitivity_Results.csv')
    df_sens = df_sens[df_sens['Algorithm'].isin(ALGOS)]
    df_sens['Success_Pct'] = df_sens['Is_Success'] * 100
except:
    df_sens = pd.DataFrame()

def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"Wygenerowano: {path}")
    plt.close()

# ==============================================================================
# 3. GENEROWANIE POPRAWIONYCH WYKRESÓW
# ==============================================================================

# --- WYKRES 1: ENERGIA ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_clean, x='Scenario_Sensors', y='Energy_Total_J', hue='Algorithm', style='Algorithm',
             palette=COLORS, markers=MARKERS, err_style="bars", errorbar=("sd", 1), 
             err_kws={'capsize': 5}) 
plt.xlabel("Liczba Sensorów")
plt.ylabel("Energia Całkowita [J]")
plt.title("Rys. 5.1. Trend zużycia energii")
plt.grid(True, alpha=0.3)
save_plot("Fig1_Energy_Trend.png")

# --- WYKRES 2: OPÓŹNIENIE (NAPRAWIONE OSIE) ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_clean, x='Scenario_Sensors', y='Delay_ms', hue='Algorithm', style='Algorithm',
             palette=COLORS, markers=MARKERS, errorbar=None) # Bez errorbar bo opóźnienie jest stałe

# WYMUSZENIE SKALI OD 0 DO 2 ms
# To sprawi, że linia będzie płaska (poprawnie), a nie poszarpana przez szum 1e-12
plt.ylim(0, 3.0) 
plt.xlabel("Liczba Sensorów")
plt.ylabel("Opóźnienie [ms]")
plt.title("Rys. 5.5. Średnie opóźnienie transmisji")
plt.grid(True, alpha=0.3)
save_plot("Fig2_Delay_Trend.png")

# --- WYKRES 3: LINK MARGIN (CZYSTSZY) ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_clean, x='Scenario_Sensors', y='Min_Link_Margin_dB', hue='Algorithm', style='Algorithm',
             palette=COLORS, markers=MARKERS, err_style="bars", errorbar=("sd", 1),
             err_kws={'capsize': 5})
plt.xlabel("Liczba Sensorów")
plt.ylabel("Link Margin [dB]")
plt.title("Rys. 5.3. Margines łącza radiowego (Link Margin)")
plt.grid(True, alpha=0.3)
save_plot("Fig4_Reliability.png")

# --- WYKRES 4: CZAS ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_clean, x='Scenario_Sensors', y='Execution_Time_s', hue='Algorithm', style='Algorithm',
             palette=COLORS, markers=MARKERS, err_style="bars", errorbar=("sd", 1),
             err_kws={'capsize': 5})
plt.xlabel("Liczba Sensorów")
plt.ylabel("Czas Obliczeń [s]")
plt.title("Rys. 5.4. Koszt obliczeniowy")
plt.grid(True, alpha=0.3)
save_plot("Fig3_Time_Cost.png")

# --- WYKRES 5: SUCCESS RATE ---
if not df_sens.empty:
    sens_agg = df_sens.groupby(['Config_Pack', 'Algorithm'])['Success_Pct'].mean().reset_index()
    labels_map = {'A_Eco': 'Eco', 'B_Standard': 'Standard', 'C_High': 'High'}
    sens_agg['Config_Label'] = sens_agg['Config_Pack'].map(labels_map)
    
    plt.figure(figsize=(9, 6))
    sns.barplot(data=sens_agg, x='Config_Label', y='Success_Pct', hue='Algorithm', 
                palette=COLORS, order=['Eco', 'Standard', 'High'], edgecolor='black')
    plt.xlabel("Zasoby")
    plt.ylabel("Skuteczność [%]")
    plt.title("Rys. 5.6. Skuteczność algorytmów (Success Rate)")
    plt.ylim(0, 110)
    save_plot("Fig6_Sensitivity.png")

# --- WYKRES 6: SŁUPKOWY ---
subset = df_clean[df_clean['Scenario_Sensors'].isin([6, 12, 20])]
plt.figure(figsize=(9, 6))
sns.barplot(data=subset, x='Scenario_Sensors', y='Energy_Total_J', hue='Algorithm',
            palette=COLORS, edgecolor='black', errorbar=None)
plt.title("Rys. 5.2. Porównanie Efektywności Energetycznej")
save_plot("Fig5_Energy_Comparison.png")

print("\n>>> SUKCES. Wykresy naprawione (Delay scale fixed, GA outliers removed).")