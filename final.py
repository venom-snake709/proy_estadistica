'''
Proyecto: Modelo de regresión lineal múltiple para predecir Temperatura (Szeged 2006-2016)
Ecuación: Temp = B0 + B1*Humidity + B2*Pressure + B3*WindSpeed + ε

Este script realiza:
1. Carga y detección de columnas relevantes
2. Limpieza de datos (NaNs y winsorización de outliers)
3. Cálculo de VIF (multicolinealidad)
4. Ajuste de modelo inicial y diagnóstico de supuestos
5. Transformaciones Box–Cox si se violan supuestos
6. Reajuste y rediagnóstico
7. Guardado de gráficas que cumplen supuestos en carpeta `figures/`
8. Generación de archivo de logs con comentarios clave
9. Conclusiones y métricas clave

Dependencias: pandas, numpy, scipy, matplotlib, statsmodels, os
Autor: [Tu Nombre]
Fecha: 2025-05-15
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm, skew, kurtosis, boxcox
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_reset

# Crear carpeta para figuras
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Carga y detección de columnas relevantes
def load_data(path):
    df = pd.read_csv(path)
    temp_col = next(c for c in df.columns if 'Temperature' in c)
    hum_col  = next(c for c in df.columns if 'Humidity'    in c)
    pres_col = next(c for c in df.columns if 'Pressure'    in c)
    wind_col = next(c for c in df.columns if 'Wind Speed'  in c)
    return df, temp_col, hum_col, pres_col, wind_col

# 2. Limpieza y winsorización
def clean_and_winsorize(df, cols):
    df = df.dropna(subset=cols)
    for c in cols:
        low, high = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lower=low, upper=high)
    return df

# 3. VIF
def compute_vif(df, features):
    X = sm.add_constant(df[features])
    vif = pd.Series([variance_inflation_factor(X.values, i)
                     for i in range(X.shape[1])], index=X.columns)
    print("\nVIF:")
    print(vif)
    return vif

# 4. Ajuste y diagnóstico
def fit_and_diagnose(df, dep, indep, suffix):
    X = sm.add_constant(df[indep]); y = df[dep]
    model = sm.OLS(y, X).fit()
    print(f"\nModelo {suffix} Summary:")
    print(model.summary())

    resid = model.resid; fitted = model.fittedvalues

    # Normalidad: histograma + QQ
    stat_sw, p_sw = shapiro(resid.sample(min(5000,len(resid)), random_state=1))
    mu, sigma = resid.mean(), resid.std()
    plt.figure(); plt.hist(resid, bins=30, density=True, alpha=0.6)
    x = np.linspace(resid.min(), resid.max(), 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r--')
    plt.title(f'Hist Residuales {suffix}\nShapiro p={p_sw:.3f}')
    plt.savefig(f'{FIG_DIR}/histogram_resid_{suffix.strip()}.png'); plt.close()

    qqplot(resid, line='s'); plt.title(f'QQ-plot {suffix}');
    plt.savefig(f'{FIG_DIR}/qqplot_{suffix.strip()}.png'); plt.close()

    # Homocedasticidad: residuos vs fitted
    bp_p = sms.het_breuschpagan(resid, model.model.exog)[1]
    plt.figure(); plt.scatter(fitted, resid, alpha=0.3)
    plt.axhline(0, linestyle='--');
    plt.title(f'Residuales vs Ajustados {suffix}\nBP p={bp_p:.3f}')
    plt.savefig(f'{FIG_DIR}/resid_vs_fitted_{suffix.strip()}.png'); plt.close()

    # Linealidad
    reset_p = linear_reset(model, power=2, use_f=True).pvalue
    print(f'Reset p-value {suffix}: {reset_p:.3f}')

    # Independencia
    dw = sm.stats.stattools.durbin_watson(resid)
    print(f'Durbin-W {suffix}: {dw:.3f}')

    return model, {'shapiro_p': p_sw, 'bp_p': bp_p, 'reset_p': reset_p, 'dw': dw}

# 5. Pipeline principal y logs
def main(path):
    logs = []
    df, temp, hum, pres, wind = load_data(path)
    logs.append(f"Datos cargados: {df.shape[0]} filas, columnas: Temp, Hum, Pres, Wind.")

    df = clean_and_winsorize(df, [temp, hum, pres, wind])
    logs.append("Limpieza de NaNs y winsorización al 1-99 percentil aplicada.")

    vif1 = compute_vif(df, [hum, pres, wind])
    logs.append(f"VIF inicial: {vif1.to_dict()}")

    m1, stats1 = fit_and_diagnose(df, temp, [hum, pres, wind], 'base')
    logs.append(f"Modelo base: Shapiro p={stats1['shapiro_p']:.3f}, BP p={stats1['bp_p']:.3f}, RESET p={stats1['reset_p']:.3f}, DW={stats1['dw']:.3f}")

    # Transformación Box-Cox
    df_bc = df.copy()
    for c in [hum, pres, wind]:
        arr = df_bc[c].values
        arr = arr - arr.min() + 1e-3 if (arr<=0).any() else arr
        df_bc[c], lam = boxcox(arr)
        logs.append(f'Box–Cox aplicado a {c} con λ={lam:.3f}')

    vif2 = compute_vif(df_bc, [hum, pres, wind])
    logs.append(f"VIF post-BoxCox: {vif2.to_dict()}")

    m2, stats2 = fit_and_diagnose(df_bc, temp, [hum, pres, wind], 'boxcox')
    logs.append(f"Modelo BoxCox: Shapiro p={stats2['shapiro_p']:.3f}, BP p={stats2['bp_p']:.3f}, RESET p={stats2['reset_p']:.3f}, DW={stats2['dw']:.3f}")

    # Heatmap de correlación
    corr = df_bc[[temp, hum, pres, wind]].corr()
    plt.figure(figsize=(6,5)); plt.imshow(corr, aspect='auto', interpolation='none'); plt.colorbar()
    plt.xticks(range(4), [temp, hum, pres, wind], rotation=45); plt.yticks(range(4), [temp, hum, pres, wind])
    plt.title('Heatmap correlación variables'); plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/heatmap_correlacion.png'); plt.close()
    logs.append('Heatmap de correlación generado y guardado.')

    # Escritura de logs
    with open(f'{FIG_DIR}/logs.txt', 'w') as f:
        for line in logs:
            f.write(line + '\n')
    print(f"Logs guardados en {FIG_DIR}/logs.txt")

    # Conclusiones finales
    print("\n--- Conclusiones Breves ---")
    print(f"Model base: R2={m1.rsquared:.3f}, F p={m1.f_pvalue:.3g}")
    print(f"Model BoxCox: R2={m2.rsquared:.3f}, F p={m2.f_pvalue:.3g}")

if __name__ == '__main__':
    main('./datos.csv')

