import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Innowacyjny Dom - Kalkulator", layout="centered")

# --- PEŁNA BAZA DANYCH ---
DANE_POMP = {
    "ACOND": {
        "GRANDIS N": {"35": {"pts": [[-20, 3.8], [-15, 4.3], [-10, 4.9], [-5, 5.7], [0, 6.6], [5, 7.4], [10, 8.1]]}, "55": {"pts": [[-20, 3.6], [-15, 4.1], [-10, 4.6], [-5, 5.3], [0, 6.0], [5, 6.7], [10, 7.4]]}},
        "GRANDIS R": {"35": {"pts": [[-20, 8.7], [-15, 10.2], [-10, 11.6], [-5, 13.0], [0, 14.8], [5, 16.2], [10, 17.8]]}, "55": {"pts": [[-20, 8.4], [-15, 9.7], [-10, 11.0], [-5, 12.4], [0, 13.7], [5, 15.2], [10, 16.6]]}}
    },
    "Mitsubishi (Zubadan)": {
        "ZUBADAN 6kW": {"35": {"pts": [[-25, 4.7], [-20, 6.0], [-15, 7.3], [-10, 8.0], [-7, 8.3], [2, 7.0], [7, 8.3]]}, "55": {"pts": [[-15, 5.8], [-10, 6.5], [-7, 6.9], [2, 6.0], [7, 6.9]], "limit": -15}},
        "ZUBADAN 10kW": {"35": {"pts": [[-25, 8.0], [-20, 9.4], [-15, 10.7], [-10, 12.0], [-7, 13.2], [2, 12.4], [7, 10.9]]}, "55": {"pts": [[-15, 9.2], [-10, 10.0], [-7, 10.9], [2, 10.4], [7, 9.2]], "limit": -15}},
        "ZUBADAN 14kW": {"35": {"pts": [[-25, 10.4], [-20, 12.0], [-15, 13.7], [-10, 15.0], [-7, 16.2], [2, 14.8], [7, 14.2]]}, "55": {"pts": [[-15, 12.5], [-10, 13.5], [-7, 14.0], [2, 13.5], [7, 12.5]], "limit": -15}}
    },
    "Mitsubishi (Eco Inverter)": {
        "ECO INVERTER 6kW": {"35": {"pts": [[-25, 2.3], [-20, 3.4], [-15, 4.3], [-10, 5.2], [-7, 6.5], [2, 5.6], [7, 6.7]]}, "55": {"pts": [[-15, 3.1], [-10, 3.6], [-7, 4.0], [2, 4.8], [7, 5.5]], "limit": -15}},
        "ECO INVERTER 10kW": {"35": {"pts": [[-25, 4.8], [-20, 6.0], [-15, 7.0], [-10, 8.0], [-7, 9.0], [2, 9.2], [7, 11.7]]}, "55": {"pts": [[-15, 5.9], [-10, 6.3], [-7, 6.8], [2, 8.5], [7, 9.5]], "limit": -15}}
    },
    "LG (R32 Mono)": {
        "MONOBLOC R32 5kW": {"35": {"pts": [[-25, 5.5], [-20, 5.5], [-15, 5.5], [-10, 5.5], [-7, 5.5], [7, 5.5]]}, "55": {"pts": [[-15, 5.0], [-7, 5.5], [7, 5.5]], "limit": -15}},
        "MONOBLOC R32 7kW": {"35": {"pts": [[-25, 5.85], [-20, 6.43], [-15, 7.0], [-7, 7.0], [7, 7.0]]}, "55": {"pts": [[-15, 6.6], [-7, 7.0], [7, 7.0]], "limit": -15}},
        "MONOBLOC R32 12kW": {"35": {"pts": [[-25, 8.5], [-20, 10.2], [-15, 12.0], [-7, 12.0], [7, 12.0]]}, "55": {"pts": [[-15, 11.5], [-7, 12.0], [7, 12.0]], "limit": -15}},
        "MONOBLOC R32 16kW": {"35": {"pts": [[-25, 12.0], [-20, 14.0], [-15, 16.0], [-7, 16.0], [7, 16.0]]}, "55": {"pts": [[-15, 15.2], [-7, 16.0], [7, 16.0]], "limit": -15}}
    },
    "LG (R290 Mono)": {
        "THERMA V R290 9kW": {"35": {"pts": [[-25, 6.4], [-20, 7.6], [-15, 8.8], [-7, 9.0], [7, 9.0]]}, "55": {"pts": [[-25, 5.1], [-20, 6.5], [-15, 7.8], [-7, 9.0], [7, 9.0]]}},
        "THERMA V R290 16kW": {"35": {"pts": [[-25, 9.0], [-20, 10.47], [-15, 12.59], [-7, 16.0], [7, 16.0]]}, "55": {"pts": [[-25, 6.79], [-20, 8.93], [-15, 11.58], [-7, 13.16], [7, 16.0]]}}
    },
    "LG Mały split": {
        "MAŁY SPLIT 4kW": {"35": {"pts": [[-20, 4.0], [-15, 4.0], [-7, 4.0], [7, 4.0]]}, "55": {"pts": [[-7, 4.0], [7, 4.0]], "limit": -7}},
        "MAŁY SPLIT 6kW": {"35": {"pts": [[-20, 6.0], [-15, 6.0], [-7, 6.0], [7, 6.0]]}, "55": {"pts": [[-7, 6.0], [7, 6.0]], "limit": -7}}
    },
    "LG split": {
        "SPLIT 7kW": {"35": {"pts": [[-20, 7.0], [-15, 7.0], [-7, 7.0], [7, 7.0]]}, "55": {"pts": [[-7, 7.0], [7, 7.0]], "limit": -7}},
        "SPLIT 9kW": {"35": {"pts": [[-20, 9.0], [-15, 9.0], [-7, 9.0], [7, 9.0]]}, "55": {"pts": [[-7, 9.0], [7, 9.0]], "limit": -7}}
    },
    "Hegam": {
        "HEGAM 6kW": {"35": {"pts": [[-25, 3.88], [-20, 4.44], [-15, 5.1], [-10, 5.87], [-7, 6.75], [7, 9.1]]}, "55": {"pts": [[-25, 3.46], [-20, 3.98], [-15, 4.57], [-10, 5.26], [-7, 6.05], [7, 8.16]]}},
        "HEGAM 16kW": {"35": {"pts": [[-25, 9.33], [-20, 10.47], [-15, 12.08], [-10, 13.92], [-7, 16.03], [7, 21.7]]}, "55": {"pts": [[-25, 8.51], [-20, 9.61], [-15, 11.02], [-10, 12.69], [-7, 14.59], [7, 19.8]]}}
    }
}

# --- 1. GÓRA: POBIERANIE DANYCH ---
st.title("🏡 Dobór Pompy Ciepła - Innowacyjny Dom")

st.subheader("⚙️ Edycja parametrów")
col_1, col_2 = st.columns(2)

with col_1:
    # Użycie on_change nie jest tu konieczne przy liniowej strukturze, 
    # ale gwarantujemy unikalne klucze
    zapotrzebowanie = st.number_input("Zapotrzebowanie przy -20°C [kW]:", 1.0, 50.0, 10.0, 0.1, key="demand")
    temp_zas = st.selectbox("Zasilanie instalacji [°C]:", ["35", "55"], key="supply")

with col_2:
    producent = st.selectbox("Producent:", sorted(list(DANE_POMP.keys())), key="mfr")
    model = st.selectbox("Model pompy:", sorted(list(DANE_POMP[producent].keys())), key="mdl")

# --- 2. OBLICZENIA (WYKONYWANE ZAWSZE PRZED WYŚWIETLENIEM) ---
try:
    konf = DANE_POMP[producent][model][temp_zas]
    t_data, p_data = zip(*konf["pts"])
    f_pompa = interp1d(t_data, p_data, kind='linear', fill_value="extrapolate")
    
    # Budynek: prosta przez (20, 0) i (-20, zapotrzebowanie)
    m_bud = -zapotrzebowanie / 40
    b_bud = zapotrzebowanie / 2
    f_budynek = lambda t: m_bud * t + b_bud

    # Szukanie przecięcia
    pb_t = float(fsolve(lambda t: f_pompa(t) - f_budynek(t), x0=-10.0)[0])
    pb_p = f_budynek(pb_t)

    # --- 3. ŚRODEK: WYŚWIETLANIE WYNIKÓW ---
    st.write("---")
    st.subheader("📍 Wynik analizy")
    
    # Wyświetlamy konfigurację pobraną prosto z widgetów
    st.markdown(f"**Konfiguracja:** :blue[{producent}] | :blue[{model}] | :blue[{temp_zas}°C]")
    
    # Wyświetlamy PB obliczony linijkę wyżej
    st.info(f"**Punkt biwalentny:** {pb_t:.2f} °C")
    
    if pb_t < -7.0:
        st.success("✅ pompa ciepła ma odpowiednio dużą moc")
    else:
        st.error("❌ pompa ciepła ma zbyt niską moc")

    if temp_zas == "55" and "limit" in konf:
        st.warning(f"⚠️ Osiągnięcie temperatury 55st.C poniżej {konf['limit']} °C zawsze wymaga grzałki.")

    # --- 4. DÓŁ: WYKRES ---
    st.write("---")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    t_os = np.linspace(-25, 20, 500)
    ax.plot(t_os, f_pompa(t_os), label=f'Moc pompy ({temp_zas}°C)', color='#1f77b4', linewidth=3)
    ax.plot(t_os, [f_budynek(t) for t in t_os], label=f'Zapotrzebowanie {zapotrzebowanie}kW', color='#d62728', linestyle='--')
    ax.scatter(pb_t, pb_p, color='black', s=150, zorder=5, label=f'PB: {pb_t:.2f}°C')
    ax.axvline(-7, color='green', linestyle=':', alpha=0.7, label='Próg -7°C')
    ax.set_xlabel('Temperatura zewnętrzna [°C]')
    ax.set_ylabel('Moc [kW]')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

except Exception:
    st.error("Brak pełnych danych dla wybranej konfiguracji. Wybierz inny model.")

