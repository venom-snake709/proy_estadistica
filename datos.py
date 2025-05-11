import pandas as pd
import os
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, kendalltau, chi2_contingency, boxcox
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# Cargar datos, incluyendo sistemas UNIX
df = pd.read_csv(r"proy_estadistica\\datos.csv") if os.name == 'nt' else pd.read_csv(r"./datos.csv")

# Filtrar solo columnas numéricas
df_numericas = df.select_dtypes(include='number')

# Intervalos de confianza
print("\n------------Intervalos de confianza-------------------")
ic_data = []
for col in df_numericas.columns:
    datos = df_numericas[col].dropna()
    stats = DescrStatsW(datos)
    ic = stats.tconfint_mean(alpha=0.05)
    ic_data.append([col, f"{ic[0]:.2f}", f"{ic[1]:.2f}"])

ic_df = pd.DataFrame(ic_data, columns=["Variable", "Límite inferior", "Límite superior"])
print(ic_df.to_string(index=False))

# Pruebas no parametricas
print("\n----------------------Pruebas no parametricas-----------------------------")
alpha = 0.05  # Nivel de significancia

# --- 1. Wilcoxon: ¿Difieren temperatura real y aparente? ---
print("\n--- Prueba de Wilcoxon ---")
df_clean = df[['Temperature (C)', 'Apparent Temperature (C)']].dropna()
stat, p_value = wilcoxon(df_clean['Temperature (C)'], df_clean['Apparent Temperature (C)'])
wilcoxon_data = [
    ["Estadístico de Wilcoxon", f"{stat:.4f}"],
    ["Valor p", f"{p_value:.4f}"],
    ["Conclusión", "Rechazamos H0." if p_value < alpha else "No rechazamos H0."]
]
wilcoxon_df = pd.DataFrame(wilcoxon_data, columns=["Métrica", "Valor"])
print(wilcoxon_df.to_string(index=False))

# --- 2. Mann-Whitney U: ¿Difieren los niveles de humedad entre 'Partly Cloudy' y 'Mostly Cloudy'? ---
print("\n--- Prueba de Mann-Whitney U ---")
group1 = df[df['Summary'] == 'Partly Cloudy']['Humidity'].dropna()
group2 = df[df['Summary'] == 'Mostly Cloudy']['Humidity'].dropna()
if not group1.empty and not group2.empty:
    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    mannwhitney_data = [
        ["Estadístico de Mann-Whitney U", f"{stat:.4f}"],
        ["Valor p", f"{p_value:.4f}"],
        ["Conclusión", "Rechazamos H0." if p_value < alpha else "No rechazamos H0."]
    ]
    mannwhitney_df = pd.DataFrame(mannwhitney_data, columns=["Métrica", "Valor"])
    print(mannwhitney_df.to_string(index=False))
else:
    print("No hay suficientes datos en ambos grupos para aplicar la prueba.")

# --- 3. Kruskal-Wallis: ¿Difiere la velocidad del viento por tipo de precipitación? ---
print("\n--- Prueba de Kruskal-Wallis ---")
rain = df[df['Precip Type'] == 'rain']['Wind Speed (km/h)'].dropna()
snow = df[df['Precip Type'] == 'snow']['Wind Speed (km/h)'].dropna()
if not rain.empty and not snow.empty:
    stat, p_value = kruskal(rain, snow)
    kruskal_data = [
        ["Estadístico de Kruskal-Wallis", f"{stat:.4f}"],
        ["Valor p", f"{p_value:.4f}"],
        ["Conclusión", "Rechazamos H0." if p_value < alpha else "No rechazamos H0."]
    ]
    kruskal_df = pd.DataFrame(kruskal_data, columns=["Métrica", "Valor"])
    print(kruskal_df.to_string(index=False))
else:
    print("No hay suficientes grupos para aplicar la prueba.")

# --- 4. Correlación de Kendall: ¿Existe relación entre temperatura y presión? ---
print("\n--- Correlación de Kendall ---")
data = df[['Temperature (C)', 'Pressure (millibars)']].dropna()
corr, p_value = kendalltau(data['Temperature (C)'], data['Pressure (millibars)'])
kendall_data = [
    ["Coeficiente de Kendall", f"{corr:.4f}"],
    ["Valor p", f"{p_value:.4f}"],
    ["Conclusión", "Rechazamos H0." if p_value < alpha else "No rechazamos H0."]
]
kendall_df = pd.DataFrame(kendall_data, columns=["Métrica", "Valor"])
print(kendall_df.to_string(index=False))

# --- 5. Chi-cuadrado: ¿Existe asociación entre tipo de precipitación y resumen del clima? ---
print("\n--- Prueba Chi-cuadrado de Independencia ---")
contingencia = pd.crosstab(df['Precip Type'], df['Summary'])
if not contingencia.empty and (contingencia.shape[0] > 1 and contingencia.shape[1] > 1):
    chi2, p_value, dof, expected = chi2_contingency(contingencia)
    chisquare_data = [
        ["Estadístico Chi-cuadrado", f"{chi2:.4f}"],
        ["Valor p", f"{p_value:.4f}"],
        ["Conclusión", "Rechazamos H0." if p_value < alpha else "No rechazamos H0."]
    ]
    chisquare_df = pd.DataFrame(chisquare_data, columns=["Métrica", "Valor"])
    print(chisquare_df.to_string(index=False))
else:
    print("No hay suficientes categorías para aplicar la prueba de Chi-cuadrado.")

# ------------------------------------
# REGRESIÓN: Temperatura ~ Presión
# ------------------------------------
print("\n--- REGRESIÓN LINEAL: Temperatura ~ Presión ---")
reg_data = df[['Temperature (C)', 'Pressure (millibars)']].dropna()
X = sm.add_constant(reg_data['Pressure (millibars)'])
y = reg_data['Temperature (C)']
modelo = sm.OLS(y, X).fit()
print(modelo.summary())

# ------------------------------------
# TRANSFORMACIÓN BOX-COX
# ------------------------------------
print("\n--- TRANSFORMACIÓN BOX-COX sobre la Temperatura ---")
y_pos = y[y > 0]  # Box-Cox solo acepta valores positivos
if not y_pos.empty:
    y_boxcox, lambda_bc = boxcox(y_pos)
    boxcox_data = [
        ["Lambda Box-Cox óptimo", f"{lambda_bc:.4f}"]
    ]
    boxcox_df = pd.DataFrame(boxcox_data, columns=["Métrica", "Valor"])
    print(boxcox_df.to_string(index=False))
else:
    print("No se pudo aplicar Box-Cox: hay valores ≤ 0")

# ------------------------------------
# ANOVA ONE-WAY: Temperatura ~ Precip Type
# ------------------------------------
print("\n--- ANOVA ONE-WAY: Temperatura ~ Precip Type ---")
anova1_data = df[['Temperature (C)', 'Precip Type']].dropna()
modelo_anova = smf.ols('Q("Temperature (C)") ~ C(Q("Precip Type"))', data=anova1_data).fit()
anova_table = sm.stats.anova_lm(modelo_anova, typ=2)
print(anova_table)

# ------------------------------------
# ANOVA TWO-WAY: Temperatura ~ Precip Type + Summary + interacción
# ------------------------------------
print("\n--- ANOVA TWO-WAY: Temperatura ~ Precip Type + Summary + interacción ---")

# Primero filtramos los datos y nos aseguramos de que haya suficientes observaciones en cada combinación
anova2_data = df[['Temperature (C)', 'Precip Type', 'Summary']].dropna()

# Verificamos las combinaciones posibles
cross_tab = pd.crosstab(anova2_data['Precip Type'], anova2_data['Summary'])
print("\nTabla de contingencia de combinaciones Precip Type x Summary:")
print(cross_tab)

# Filtramos solo las combinaciones con suficientes datos (por ejemplo, al menos 5 observaciones)
valid_combinations = cross_tab > 5
valid_precip = valid_combinations.any(axis=1)
valid_summary = valid_combinations.any(axis=0)

# Aplicamos el filtro
anova2_data_filtered = anova2_data[
    anova2_data['Precip Type'].isin(valid_precip[valid_precip].index) & 
    anova2_data['Summary'].isin(valid_summary[valid_summary].index)
]

if not anova2_data_filtered.empty:
    print("\nCombinaciones válidas después del filtrado:")
    print(pd.crosstab(anova2_data_filtered['Precip Type'], anova2_data_filtered['Summary']))
    
    # Modelo sin interacción primero para verificar
    print("\n--- Modelo sin interacción ---")
    modelo_anova2a = smf.ols('Q("Temperature (C)") ~ C(Q("Precip Type")) + C(Q("Summary"))', 
                           data=anova2_data_filtered).fit()
    print(sm.stats.anova_lm(modelo_anova2a, typ=2))
    
    # Modelo con interacción solo si hay suficientes datos
    if len(valid_precip[valid_precip]) > 1 and len(valid_summary[valid_summary]) > 1:
        print("\n--- Modelo con interacción ---")
        modelo_anova2b = smf.ols('Q("Temperature (C)") ~ C(Q("Precip Type")) + C(Q("Summary")) + C(Q("Precip Type")):C(Q("Summary"))', 
                               data=anova2_data_filtered).fit()
        try:
            anova_table2 = sm.stats.anova_lm(modelo_anova2b, typ=2)
            print(anova_table2)
        except:
            print("\nNo se pudo calcular el ANOVA con interacción debido a problemas de rango en la matriz de diseño")
    else:
        print("\nNo hay suficientes combinaciones para analizar la interacción")
else:
    print("\nNo hay suficientes datos después del filtrado para realizar el ANOVA de dos vías")
