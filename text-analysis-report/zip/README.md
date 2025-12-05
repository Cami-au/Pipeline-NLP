# 🧠 Text Analysis Report — NLP Pipeline Profesional

Este proyecto implementa un **pipeline completo de NLP** capaz de:

- procesar texto desde archivos CSV,
- generar visualizaciones,
- analizar tópicos mediante LDA,
- detectar anomalías con métodos avanzados,
- producir un **reporte web interactivo**,
- y permitir configuración dinámica vía CLI + JSON.

Está diseñado para trabajar con **cualquier dataset** que contenga una columna de texto.


---

# 📌 Tabla de Contenidos

1. [Características principales](#-características-principales)
2. [Requisitos](#-requisitos)
3. [Instalación](#-instalación)
4. [Estructura del proyecto](#-estructura-del-proyecto)
5. [Uso del script (CLI)](#-uso-del-script-cli)
6. [Configuración (settings.json)](#-configuración-settingsjson)
7. [Descripción del pipeline](#-descripción-del-pipeline)
8. [Reporte Interactivo](#-reporte-interactivo)
9. [Extensiones futuras](#-extensiones-futuras)
10. [Licencia](#-licencia)


---

# 🌟 Características principales

Este pipeline realiza:

### ✔ Entrada de datos
- Lectura de CSV
- Selección de la columna de texto

### ✔ Preprocesamiento robusto
- Normalización
- Eliminación de emojis
- Acentos opcional
- Eliminación de puntuación
- Tokenización
- Stopwords
- Lematización
- Compatibilidad español/inglés

### ✔ Visualizaciones
- Nube de palabras
- Top palabras
- Top-10 bigramas y trigramas
- Gráficos configurables (paletas accesibles)

### ✔ Modelado de tópicos (LDA)
- Palabras clave por tópico
- Documento representativo
- Ablación de tópicos (palabras únicas por tópico)

### ✔ Visualización avanzada
- UMAP interactivo coloreado por tópicos
- UMAP de outliers

### ✔ Detección de outliers
- Topic Confidence
- UMAP z-score
- DBSCAN
- Isolation Forest
- One-Class SVM
- Sistema **ponderado** de decisión
- Razón del outlier + score explicativo

### ✔ Reporte Web Interactivo
- Plantilla Jinja2 profesional
- Visualizaciones incrustadas
- Tablas dinámicas
- Iframes de Altair

### ✔ CLI profesional
Permite:

--file
--textcol
--topics
--title
--config



---

# ⚙️ Requisitos

Python 3.10+

Instalar dependencias:


Librerías clave:
- pandas
- numpy
- nltk
- spacy
- scikit-learn
- umap-learn
- altair
- wordcloud
- jinja2


---

# 📁 Estructura del proyecto

