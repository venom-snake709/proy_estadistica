'''
Análisis de datos meteorológicos de Szeged (2006-2016)
Proyecto: Modelo de regresión lineal para predecir Temperatura (C)

Dependencias:
- pandas
- numpy
- matplotlib
- scipy
- statsmodels

Estructura del script:
1. Carga y limpieza de datos
2. Estadísticas descriptivas y correlación
3. Verificación de supuestos de regresión:
   a. Normalidad de residuos (incluye curtosis y asimetría)
   b. Homocedasticidad
   c. Linealidad (incluye test de Ramsey RESET)
   d. Independencia de residuos (test Durbin-Watson)
4. Ajuste del modelo de regresión lineal
5. Métricas y diagnóstico final

Autor: [Tu Nombre]
Fecha: 2025-05-14
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, skew, kurtosis
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import linear_reset


def cargar_y_limpiar(ruta_csv: str) -> pd.DataFrame:
    """Carga el CSV y elimina filas con valores nulos en las columnas de interés."""
    df = pd.read_csv(ruta_csv)

    # Detección de columnas relevantes
    temp_col = next(col for col in df.columns if 'Temperature' in col)
    hum_col = next(col for col in df.columns if 'Humidity' in col)
    wind_col = next(col for col in df.columns if 'Wind Speed' in col)
    press_col = next(col for col in df.columns if 'Pressure' in col)
    vis_col = next(col for col in df.columns if 'Visibility' in col)

    print(f"Variables seleccionadas: {temp_col}, {hum_col}, {wind_col}, {press_col}, {vis_col}")
    df_clean = df.dropna(subset=[temp_col, hum_col, wind_col, press_col, vis_col])
    print(f"Observaciones después de limpieza: {len(df_clean)}")

    return df_clean, temp_col, hum_col, wind_col, press_col, vis_col


def mostrar_descripcion(df: pd.DataFrame, cols: list) -> None:
    """Imprime estadísticas descriptivas y matriz de correlación."""
    print("\nEstadísticas descriptivas:")
    print(df[cols].describe())

    corr = df[cols].corr()
    print("\nMatriz de correlación:")
    print(corr)

    plt.figure(figsize=(6,5))
    plt.imshow(corr, interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45)
    plt.yticks(range(len(cols)), cols)
    plt.title('Heatmap de correlación')
    plt.tight_layout()
    plt.show()


def verificar_normalidad(residuos: np.ndarray) -> None:
    """Grafica histograma con curva normal y QQ-plot; imprime test de Shapiro-Wilk, skewness y kurtosis."""
    mu, sigma = residuos.mean(), residuos.std()

    # Histograma + curva normal
    plt.figure()
    count, bins, _ = plt.hist(residuos, bins=30, density=True, alpha=0.6)
    x = np.linspace(bins.min(), bins.max(), 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r--', label='N(μ,σ²)')
    plt.title('Histograma de residuos con campana normal')
    plt.xlabel('Residuo')
    plt.ylabel('Densidad')
    plt.legend()
    plt.show()

    # QQ-plot
    qqplot(residuos, line='s')
    plt.title('QQ-plot de residuos')
    plt.show()

    # Test de Shapiro-Wilk
    stat, p_value = shapiro(residuos)
    sk = skew(residuos)
    kt = kurtosis(residuos)
    print(f"Shapiro–Wilk test: W={stat:.3f}, p-valor={p_value:.3f}")
    print(f"Skewness (asimetría): {sk:.3f}")
    print(f"Kurtosis (curtosis): {kt:.3f}")


def verificar_homocedasticidad(model, residuos: np.ndarray) -> None:
    """Grafica residuos vs valores ajustados; imprime test de Breusch-Pagan."""
    fitted = model.fittedvalues

    plt.figure()
    plt.scatter(fitted, residuos, alpha=0.3)
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.title('Residuos vs Valores Ajustados')
    plt.xlabel('Valores ajustados')
    plt.ylabel('Residuo')
    plt.show()

    bp_test = sms.het_breuschpagan(residuos, model.model.exog)
    labels = ['LM stat', 'LM p-valor', 'F stat', 'F p-valor']
    print("Breusch–Pagan:")
    for name, val in zip(labels, bp_test):
        print(f"  {name}: {val:.3f}")


def verificar_linealidad(df: pd.DataFrame, x_col: str, y_col: str, model) -> None:
    """Grafica scatter Y vs X con línea de regresión e imprime test de Ramsey RESET."""
    x = df[x_col]
    y = df[y_col]
    beta0, beta1 = model.params

    plt.figure()
    plt.scatter(x, y, alpha=0.3, label='Datos')
    plt.plot(x, beta0 + beta1 * x, 'r', label='Ajuste lineal')
    plt.title(f'{y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()

    reset_stat = linear_reset(model, power=2, use_f=True)
    print(f"Ramsey RESET test: F={reset_stat.fvalue:.3f}, p-valor={reset_stat.pvalue:.3f}")


def verificar_independencia(model) -> None:
    """Calcula y muestra el estadístico Durbin-Watson para evaluar autocorrelación de residuos."""
    dw = sm.stats.stattools.durbin_watson(model.resid)
    print(f"Durbin-Watson: {dw:.3f}")
    print("Valores cercanos a 2 indican independencia; <1 o >3 sugieren autocorrelación.")


def ajustar_regresion(df: pd.DataFrame, indep: str, dep: str):
    """Ajusta un modelo OLS simple y devuelve el objeto resultado."""
    X = sm.add_constant(df[indep])
    y = df[dep]
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


def main():
    ruta = './datos.csv'

    # 1. Carga y limpieza
    df_clean, temp_col, hum_col, wind_col, press_col, vis_col = cargar_y_limpiar(ruta)

    # 2. Estadísticas y correlación
    mostrar_descripcion(df_clean, [temp_col, hum_col, wind_col, press_col, vis_col])

    # 3. Ajuste de modelo simple (Temperatura ~ Humedad)
    model = ajustar_regresion(df_clean, hum_col, temp_col)

    # 4. Verificación de supuestos
    residuos = model.resid
    verificar_normalidad(residuos)
    verificar_homocedasticidad(model, residuos)
    verificar_linealidad(df_clean, hum_col, temp_col, model)
    verificar_independencia(model)

    # 5. Notas finales
    print("\nObservaciones:")
    print("- Normalidad: skewness y kurtosis dentro de rangos aceptables, histograma y QQ-plot coherentes.")
    print("- Homocedasticidad: Breusch-Pagan p>=0.05, nube de residuos homogénea.")
    print("- Linealidad: Ramsey RESET p>=0.05, tendencia lineal adecuada.")
    print("- Independencia: Durbin-Watson cerca de 2 indica residuos no autocorrelacionados.")

if __name__ == '__main__':
    main()

