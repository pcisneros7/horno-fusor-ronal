# Reto: Horno Fusor de Aluminio
## Modelado y control de temperatura mediante control clásico, moderno y predictores basados en redes neuronales

**Equipo:** Sara Garza Reyna, Ana Paulina Cisneros Negrete, María José Zamora Gómez, Santiago Castellanos Saavedra
**Materia:** Análisis numérico para la optimización no-lineal — Grupo 601
**Empresa colaboradora:** RONAL Group

---

## 1. Descripción general

Este proyecto analiza la dinámica térmica de un horno fusor de aluminio (referencia: Stack Tower Melter, Lindberg/MPH) operando a una temperatura objetivo de 780 °C. Se desarrollan y comparan dos enfoques para modelar y controlar la temperatura del horno:

1. **Control clásico (Laplace / Lugar Geométrico de las Raíces):** se obtiene una función de transferencia de primer orden con retardo (FOPDT), se aproxima el retardo con Padé de primer orden, y se sintoniza un controlador PID utilizando el Teorema del Valor Final para corregir el error en estado estable.

2. **Modelado basado en datos (Deep Learning):** se realiza un análisis exploratorio (EDA) y un análisis de clustering (K-Means + PCA) sobre datos reales de operación del horno, se entrena un modelo preliminar con Prophet, un modelo LSTM (benchmark) y finalmente una **CNN-1D** que logra aproximar la dinámica del sistema. A partir de la respuesta de la CNN ante un escalón se ajusta un modelo FOPDT y se obtiene una función de transferencia comparable con el modelo clásico.

Ambos modelos se comparan en términos de polos, ceros, estabilidad, error en estado estable y respuesta ante distintas entradas (escalón, ruido aleatorio, señal senoidal).

---

## 2. Estructura del proyecto

```
.
├── README.md
├── requirements.txt
├── dataset_horno_fusor2.csv           # Dataset proporcionado por RONAL Group (21,600 muestras)
├── EDA_horno_fusor.ipynb              # Análisis exploratorio de datos
├── clustering_FINAL_horno.ipynb       # PCA + K-Means: regímenes de operación del horno
├── lstm_horno.ipynb                   # Modelo preliminar de Deep Learning (LSTM) y ajuste FOPDT
└── prophet_modelo_control__3_.ipynb   # Modelo preliminar Prophet y extracción de coeficientes
```

> **Nota:** la implementación de la CNN-1D, el diagrama de bloques en Simulink y las simulaciones de control clásico/PID se describen en el informe (`Reto_final_control.pdf`), pero no se incluyen como notebooks en este repositorio. Si se generan los archivos correspondientes (p. ej. `cnn_horno.ipynb`, `control_clasico.slx` o equivalentes en Python/`control`), deberán colocarse en la raíz del proyecto y agregarse aquí.

### 2.1 Dataset

El archivo `dataset_horno_fusor2.csv` se incluye en la raíz del proyecto (mismo nivel que los notebooks) y contiene 21,600 muestras con las siguientes columnas:

| Columna | Descripción |
|---|---|
| `timestamp` | Marca de tiempo de la muestra |
| `sensor_temp` | Temperatura medida por el sensor (°C) |
| `setpoint` | Temperatura de referencia (°C) |
| `gas_flow` | Flujo de gas (%) |
| `air_flow` | Flujo de aire |
| `furnace_load` | Carga de aluminio (kg) |
| `ambient_temp` | Temperatura ambiente (°C) |
| `gas_pressure` | Presión de gas |
| `energy_consumption` | Consumo energético |

Variables derivadas para el análisis: `error` (`sensor_temp - setpoint`) y `turno` (derivado de `timestamp`).

---

## 3. Requisitos e instalación

### 3.1 Requisitos previos
- Python 3.10 o superior
- pip
- (Opcional) un entorno virtual (`venv` o `conda`)

### 3.2 Instalación

```bash
# 1. Clonar / descargar el proyecto y entrar a la carpeta
cd reto-horno-fusor

# 2. Crear y activar un entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota sobre Prophet:** en algunos sistemas (especialmente Windows) `prophet` requiere `cmdstanpy` y un compilador C++. Si la instalación falla, instalar primero:
> ```bash
> pip install cmdstanpy
> python -m cmdstanpy.install_cmdstan
> ```

> **Nota sobre PyTorch:** la versión instalada por defecto via `pip install torch` es para CPU. Si se cuenta con GPU NVIDIA y se desea acelerar el entrenamiento de la CNN/LSTM, instalar la versión correspondiente desde [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 4. Instrucciones de ejecución

Se recomienda ejecutar los notebooks en el siguiente orden, ya que cada etapa depende de los hallazgos de la anterior:

1. **`EDA_horno_fusor.ipynb`**
   Carga `dataset_horno_fusor2.csv`, genera estadísticas descriptivas, distribuciones de variables, matriz de correlaciones y un análisis preliminar por turno.

2. **`clustering_FINAL_horno.ipynb`**
   Aplica PCA (3 componentes, ~85% de varianza explicada) y K-Means (5 clusters) para identificar regímenes de operación del horno (cargas baja/media/alta, estabilización y régimen de alto error).

3. **`prophet_modelo_control__3_.ipynb`**
   Entrena un modelo Prophet univariado y con regresores (`air_flow`, `furnace_load`) para predecir `sensor_temp`, usado como baseline antes de pasar a Deep Learning.

4. **`lstm_horno.ipynb`**
   Entrena un modelo LSTM como benchmark preliminar, simula la respuesta al escalón y ajusta un modelo FOPDT (resultados no realistas, usados únicamente como punto de comparación).

Para ejecutar:

```bash
jupyter notebook
```

y abrir cada notebook en el orden indicado, ejecutando todas las celdas (`Run All`).

---

## 5. Descripción de los experimentos y resultados principales

### 5.1 Análisis exploratorio y clustering
- Error promedio entre setpoint y temperatura medida: **2.77 °C**, con picos de hasta **+40 °C / -50 °C** (1.7% de muestras fuera de rango).
- `furnace_load` presenta una distribución bimodal, indicando dos regímenes de carga.
- PCA: 3 componentes principales explican ~85% de la varianza. PC1 se interpreta como energía/variables internas del horno; PC2 como ajustes del operador/turno.
- K-Means (5 clusters) identifica regímenes de carga baja/media/alta, un régimen de estabilización (cluster 2) y un régimen de alto error (cluster 4).

### 5.2 Modelo preliminar (Prophet)
- Captura adecuadamente la tendencia general y los cambios de setpoint, pero no es suficiente para extraer una función de transferencia (K, τ, L).

### 5.3 Modelo preliminar de Deep Learning (LSTM)
- RMSE: 4.84 °C, MAE: 3.01 °C.
- Parámetros FOPDT obtenidos no son físicamente realistas (τ = 16.4 s, L = 9.4 s), por lo que se usa solo como benchmark.

### 5.4 Modelo CNN-1D (modelo final basado en datos)
- Arquitectura: 2 capas Conv1D (5→64→128 canales, kernel 3) + AvgPool1D + capa fully connected (Dropout 0.2).
- Ventana de entrada (lookback): 900 pasos (5 horas), consistente con la constante de tiempo teórica del horno (τ ≈ 3 h).
- Métricas: **RMSE = 0.6019 °C, MAE = 0.4009 °C, R² = 0.9893**.
- Parámetros FOPDT extraídos (con τ fijado al valor teórico = 10,000 s):
  - K = 0.2319 °C/unidad de gas
  - τ = 10,000 s (166.67 min)
  - L = 69.92 s (1.17 min)

### 5.5 Modelo de control clásico (PID + LGR)
- Función de transferencia de la planta: `G(s) = 780·e^(-600s) / (10800s + 1)`.
- Diseño del PID mediante Lugar Geométrico de las Raíces, con polos deseados definidos por ζ = 0.7071 y ωₙ ≥ 1.
- Corrección de error en estado estable (de Tss = 735°C a Tss ≈ 779.67°C) mediante el Teorema del Valor Final (ganancia ≈ 1.06).
- Sistema en lazo cerrado: polo real dominante y par complejo conjugado → respuesta subamortiguada con sobreimpulso.

### 5.6 Comparativa de modelos
| | Modelo clásico (PID) | Modelo CNN (datos) |
|---|---|---|
| Polos | 1 real + par complejo conjugado | 3 reales |
| Comportamiento | Subamortiguado, con sobreimpulso y oscilaciones | Sobreamortiguado, respuesta suave y lenta |
| Cero en semiplano derecho | sí (+0.0033, fase no mínima) | sí (+0.0129, fase no mínima) |
| e_ss | ≈ 0.33 °C | ≈ 0.00 °C |

**Conclusión principal:** ambos enfoques son complementarios. El modelo clásico es útil para el diseño y sintonización del controlador PID (interpretabilidad matemática), mientras que el modelo CNN representa con mayor fidelidad el comportamiento real del horno a partir de datos históricos, pero con menor interpretabilidad y mayor dependencia de la calidad de los datos de entrenamiento.

---

## 6. Reproducibilidad

- Las semillas aleatorias (`random_state` / `seed`) utilizadas en K-Means, en la división train/test y en la inicialización de los modelos de PyTorch deben fijarse explícitamente al inicio de cada notebook para garantizar resultados idénticos entre ejecuciones.
- El checkpoint del mejor modelo CNN se guarda como `best_cnn.pt` (generado durante la ejecución de `lstm_horno.ipynb` o del notebook de la CNN, según corresponda).
- División del dataset: 80% entrenamiento / 20% prueba, preservando el orden temporal (sin mezclar).

---

## 7. Referencias

1. Franklin, G., et al. (2020). *Feedback Control of Dynamic Systems*. Pearson.
2. Lindberg/MPH. (2026). *Stack (Tower) Melter*. https://www.lindbergmph.com/products/non-ferrous-melting-and-holding-furnaces/aluminum-melting-and-holding-furnaces/stack-tower-melter/
3. Ogata, K. (2010). *Ingeniería de Control Moderna*. Pearson Educación.
4. Ronal Group. (2024). *Historia*. https://www.ronalgroup.com/es/historia/
5. Smith, G. M. (2024). *¿Qué es un controlador PID?* DeweSOFT. https://dewesoft.com/es/blog/que-es-un-controlador-pid
6. Uspensky, J. V. (2005). *Teoría de Ecuaciones*. Limusa.
7. Wang, J. (2026). *Introducción a la memoria a Corto-Largo Plazo (LSTM)*. MathWorks. https://la.mathworks.com/discovery/lstm.html
8. Zill, D. (2009). *Ecuaciones diferenciales con aplicaciones de modelado*. Brooks & Cole Cengage.
