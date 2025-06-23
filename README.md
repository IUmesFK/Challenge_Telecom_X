# 📊 Análisis de Evasión de Clientes - TelecomX

Este proyecto tiene como objetivo analizar los motivos por los cuales los clientes abandonan la empresa de telecomunicaciones *Telecom X*, buscando patrones que permitan entender el comportamiento de deserción y ayudar a la empresa a reducir su tasa de abandono.

---

## 🧠 Contexto

La pérdida de clientes (churn) es un indicador crítico que puede afectar gravemente los ingresos y el crecimiento de una empresa. Telecom X detectó una tasa significativa de deserción, por lo que este análisis se centra en estudiar:

- Características demográficas de los clientes
- Tipos de contratos y servicios contratados
- Métodos de pago y gasto total
- Relación entre antigüedad y probabilidad de abandono

---

## 🔍 Proceso de análisis

1. **Extracción de datos**  
   Se cargan los datos desde un archivo JSON alojado en GitHub.

2. **Limpieza y normalización**  
   - Conversión de columnas a tipos adecuados (`bool`, `int`, `str`)
   - Identificación y tratamiento de valores nulos
   - Renombramiento de variables para mayor claridad

3. **Análisis exploratorio (EDA)**  
   - Gráficos de barras, tortas y distribuciones
   - Estudio de correlaciones entre características y churn
   - Identificación de variables relevantes

4. **Visualización**  
   Se emplean herramientas como **Matplotlib** y **Seaborn** para representar de forma clara los patrones encontrados.

---

## 🛠️ Tecnologías utilizadas

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📌 Notas adicionales
El dataset utilizado fue provisto por Alura Latam.

No se incluyen modelos predictivos; el foco está en la exploración y visualización.
