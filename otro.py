'''
Análisis de datos meteorológicos de Szeged (2006-2016)
Proyecto: Modelo de regresión lineal para predecir Temperatura (C)

Dependencias:
- pandas
- numpy
- matplotlib
- scipy
- statsmodels
- logging

Estructura del script:
1. Carga y limpieza de datos
2. Estadísticas descriptivas y correlación
3. Ajuste y métricas principales (R², R² ajustado, F-stat)
4. Transformación Box–Cox si no se cumplen supuestos
5. Verificación de supuestos
6. Guardado de figuras y resultados
7. Logging detallado

Autor: [Tu Nombre]
Fecha: 2025-05-14
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, shapiro, skew, kurtosis
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import linear_reset
import logging
from scipy.stats import norm
import os

# Configuración de logging
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/regresion.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info('Inicio de análisis de regresión')

# Guardado de figura
def save_fig(fig, name):
    os.makedirs('figures', exist_ok=True)
    path = os.path.join('figures', f"{name}.png")
    fig.savefig(path, bbox_inches='tight')
    logging.info(f"Figura guardada: {path}")

# 1. Carga y limpieza de datos
def cargar_y_limpiar(ruta_csv):
    df = pd.read_csv(ruta_csv)
    # Columnas relevantes
    temp_col = next(c for c in df.columns if 'Temperature' in c)
    hum_col = next(c for c in df.columns if 'Humidity' in c)
    df_clean = df.dropna(subset=[temp_col, hum_col])
    logging.info(f"Cargados {len(df)} registros, quedan {len(df_clean)} tras limpiar nulos")
    return df_clean, temp_col, hum_col

# 2. Estadísticas descriptivas y correlación
def descripcion(df, cols):
    desc = df[cols].describe()
    corr = df[cols].corr()
    logging.info('Estadísticas descriptivas:\n%s', desc.to_string())
    logging.info('Matriz de correlación:\n%s', corr.to_string())
    # Heatmap
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.imshow(corr, aspect='auto')
    plt.colorbar(cax, ax=ax)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45)
    ax.set_yticklabels(cols)
    ax.set_title('Heatmap de correlación')
    save_fig(fig, 'heatmap_correlacion')

# 3. Ajuste y métricas principales
def ajustar_y_metricas(df, indep, dep):
    X = sm.add_constant(df[indep])
    y = df[dep]
    model = sm.OLS(y, X).fit()
    # Métricas explícitas
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    f_stat = model.fvalue
    f_pval = model.f_pvalue
    logging.info(f"R2={r2:.4f}, R2 ajustado={r2_adj:.4f}, F={f_stat:.2f}, p-val F={f_pval:.3g}")
    print(f"R2: {r2:.4f}, R2 ajustado: {r2_adj:.4f}")
    print(f"F-estadístico: {f_stat:.2f}, p-valor: {f_pval:.3g}")
    return model

# 4. Transformación Box–Cox
def aplicar_boxcox(df, dep):
    y = df[dep].values
    # Asegurar positividad
    if np.any(y <= 0):
        y = y - y.min() + 1e-3
        logging.info('Desplazado y para Box-Cox (positividad)')
    y_box, lam = boxcox(y)
    df[f'{dep}_boxcox'] = y_box
    logging.info(f"Box-Cox aplicado con λ={lam:.4f}")
    return df, lam

# 5. Verificación de supuestos (normalidad, homocedasticidad, linealidad, independencia)
def verificar_supuestos(df, indep, dep, model):
    residuos = model.resid
    # Normalidad
    stat, p = shapiro(residuos)
    sk = skew(residuos)
    kt = kurtosis(residuos)
    logging.info(f"Shapiro W={stat:.3f}, p={p:.3f}, skew={sk:.3f}, kurtosis={kt:.3f}")
    # Gráficos
    fig1 = plt.figure(); plt.hist(residuos, bins=30, density=True, alpha=0.6);
    x = np.linspace(residuos.min(), residuos.max(), 100)
    plt.plot(x, norm.pdf(x, residuos.mean(), residuos.std()), 'r--')
    plt.title('Histograma residuos con normal')
    save_fig(fig1, 'hist_residuos')
    fig2 = qqplot(residuos, line='s'); save_fig(plt.gcf(), 'qqplot_residuos')
    # Homocedasticidad
    fig3 = plt.figure(); plt.scatter(model.fittedvalues, residuos, alpha=0.3);
    plt.axhline(0, linestyle='--'); plt.title('Residuos vs Ajustados');
    save_fig(fig3, 'residuos_vs_ajustados')
    bp = sms.het_breuschpagan(residuos, model.model.exog)
    logging.info(f"Breusch-Pagan p={bp[1]:.3f}")
    # Linealidad (Ramsey RESET)
    reset = linear_reset(model, power=2, use_f=True)
    logging.info(f"Ramsey RESET F={reset.fvalue:.3f}, p={reset.pvalue:.3f}")
    # Independencia
    dw = sm.stats.stattools.durbin_watson(residuos)
    logging.info(f"Durbin-Watson={dw:.3f}")

# 6. Main
def main():
    setup_logging()
    ruta = 'datos.csv'
    df, temp_col, hum_col = cargar_y_limpiar(ruta)
    descripcion(df, [temp_col, hum_col])
    # Modelo original
    model = ajustar_y_metricas(df, hum_col, temp_col)
    verificar_supuestos(df, hum_col, temp_col, model)
    # Si falla supuestos, probar Box–Cox
    df_box, lam = aplicar_boxcox(df.copy(), temp_col)
    model_box = ajustar_y_metricas(df_box, hum_col, f'{temp_col}_boxcox')
    verificar_supuestos(df_box, hum_col, f'{temp_col}_boxcox', model_box)
    logging.info('Análisis completado')

if __name__ == '__main__':
    main()

