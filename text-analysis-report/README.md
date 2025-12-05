# Text Analysis Report — NLP by Camila Ramírez

Este proyecto implementa un **pipeline completo de NLP** capaz de:

- procesar texto desde archivos CSV,
- generar visualizaciones,
- analizar tópicos mediante LDA,
- detectar anomalías con métodos avanzados,
- producir un **reporte web interactivo**,
- y permitir configuración dinámica vía CLI + JSON.

Está diseñado para trabajar con **cualquier dataset** que contenga una columna de texto y permite extender fácilmente sus componentes internos.

---
- Extra.
**Para usar un csv, se debe agregar en la carpeta "data" para su uso.**
**Si es mas de un csv, se añaden ambos y en settings.json se indica cual se analizará primero**

---

# Tabla de Contenidos

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

# Características principales

Este pipeline realiza:

### ✔ Entrada de datos
- Lectura de CSV
- Selección de la columna de texto
- Validación de estructura mínima del dataset.

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
Se generan gráficas interactivas mediante Altair:
- Nube de palabras
- Frecuencia de palabras
- Top palabras
- Top-10 bigramas y trigramas
- Gráficos configurables (paletas accesibles)

### ✔ Modelado de tópicos (LDA/BERTopic)
El pipeline construye y analiza tópicos mediante:
- LDA (rápido y ligero)
- BERTopic (basado en embeddings + clustering avanzado)

Incluye:
- Palabras clave por tópico
- Documento representativo
- Distribución de documentos por tópico
- Identificación del tópico dominante
- Ablación de términos únicos por tópico

### ✔ Visualización avanzada
- UMAP interactivo coloreado por tópicos
- UMAP de outliers
- Soporte para distintas paletas y formas de puntos

### ✔ Detección de outliers
Múltiples métodos incluidos:
- Topic Confidence
- UMAP z-score
- DBSCAN
- Isolation Forest
- One-Class SVM

Adicional:
- Votación **ponderada** entre métodos
- Razón del outlier + score explicativo

### ✔ Reporte Web Interactivo
Construido con Jinja2:
- Plantilla profesional
- Visualizaciones incrustadas mediante iframes
- Tablas dinámicas
- Secciones explicativas generadas automáticamente
- Exportación directa a /data/outputs/

### ✔ CLI profesional
Permite:

--file          Ruta del CSV
--textcol       Columna de texto
--topics        Número de tópicos
--title         Título del reporte HTML
--topic-method  lda | bertopic
--palette-id    Paleta visual (1–5)

---

# Requisitos

Python 3.10+

Instalación estándar:
- pip install -r requirements.txt
- python -m spacy download es_core_news_sm
- python -m nltk.downloader stopwords punkt


---

# Estructura del proyecto
project/
│
├── main.py
├── README.md
├── requirements.txt
│
├── codigo_fuente/
│   ├── pipeline/
│   ├── preprocessing/
│   ├── visualization/
│   ├── topics/
│   └── config/
│
├── data/
│   ├── *.csv
│   └── outputs/
│
├── templates/
└── static/

---

# Descripción técnica por carpeta

1. main.py — Archivo ejecutable

- Punto de entrada del sistema.
- Implementa el CLI oficial.

Gestiona:
- detección automática del CSV
- parámetros del usuario
- ejecución del pipeline completo

✔ Se puede modificar
✔ Se debe ejecutar desde consola


2. codigo_fuente/ — Núcleo del proyecto
pipeline/

Contiene **NLPReportPipeline**- , responsable de:

- flujo completo del análisis
- integración entre módulos
- orquestación de procesamiento, viz, tópicos y reporte

Modificable solo si quieres alterar el orden del pipeline.


preprocessing/

Incluye todo el procesamiento del texto:
- normalización
- tokenización
- stopwords
- lematización

Para personalizar limpieza de texto.


visualization/

Genera todas las visualizaciones:
- Wordcloud
- Histogramas
- Bigrams / Trigrams
- UMAP (tópicos y outliers)

Modificable si necesitas ajustes visuales o parámetros UMAP.


topics/

Implementación de:
- LDA
- BERTopic
- extracción de keywords
- asignación de tópico dominante

Modificable si quieres mejorar el modelado o añadir nuevos algoritmos.



config/

settings.json: configuración editable por usuario

config_loader.py: carga segura de parámetros

✔ Modificable: settings.json
⚠ No recomendado modificar config_loader.py

3. data/

*.csv: datasets de entrada

outputs/: resultados exportados (UMAPs, tablas, reporte HTML)

✔ Puedes modificar estos archivos libremente.

4. templates/

Plantillas HTML del reporte.

✔ Modificables para personalizar el diseño del dashboard.

5. static/

Recursos estáticos:
- CSS
- JS
- imágenes

Permite personalizar completamente la apariencia del reporte.

---
# Uso del script (CLI)

- Ejecución mínima
**python main.py --file data/dataset.csv**

- Especificar columna de texto
**python main.py --textcol comentario**

- Cambiar número de tópicos
**python main.py --topics 8**

- Forzar método de tópicos
**python main.py --topic-method bertopic**

- Cambiar título del reporte
**python main.py --title "Análisis Opiniones 2025"**

---
# Descripción del pipeline

Etapas internas:
- Carga y validación del CSV
- Normalización y limpieza del texto
- Tokenización + lematización
- Vectorización / embeddings
- Modelado de tópicos (LDA/BERTOPIC)
- Cálculo del tópico dominante
- Generación del UMAP
- Detección de outliers
- Construcción del reporte HTML
- Exportación a carpeta data/outputs/

---
# Reporte Interactivo

Incluye:
- Nube de palabras
- Distribución de tokens
- Tópicos explicados
- Documentos representativos
- UMAP de tópicos
- UMAP de outliers
- Tablas explicativas
- Resumen del procesamiento y parámetros usados

Totalmente navegable desde cualquier navegador moderno.

---
