import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import wilcoxon

# Cargar datos
df = pd.read_csv(r"proy_estadistica\\datos.csv")

# Filtrar solo columnas numéricas
df_numericas = df.select_dtypes(include='number')

# Intervalos de confianza
print("\n------------Intervalos de confianza-------------------")
for col in df_numericas.columns:
    datos = df_numericas[col].dropna()
    stats = DescrStatsW(datos)
    ic = stats.tconfint_mean(alpha=0.05)
    print(f"{col}: ({ic[0]:.2f}, {ic[1]:.2f})")


#Pruebas no parametricas
print("\n----------------------Pruebas no parametricas-----------------------------")
df_clean = df[['Temperature (C)', 'Apparent Temperature (C)']].dropna()

# Aplicar la prueba de Wilcoxon
stat, p_value = wilcoxon(
    df_clean['Temperature (C)'],         # Muestra 1: Temperatura real
    df_clean['Apparent Temperature (C)'] # Muestra 2: Sensación térmica
)

# Imprimir resultados
print(f'Estadístico de Wilcoxon: {stat:.4f}')
print(f'Valor p: {p_value:.4f}')

# Interpretación básica
alpha = 0.05
if p_value < alpha:
    print("Rechazamos H0: Existen diferencias significativas entre la temperatura real y la sensación térmica.")
else:
    print("No rechazamos H0: No hay evidencias de diferencias significativas.")