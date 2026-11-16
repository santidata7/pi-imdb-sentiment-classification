# Pi | Clasificación de Sentimientos en Reseñas de IMDb mediante Procesamiento de Lenguaje Natural
## Descripción del Proyecto

**Film Junky Union** busca desarrollar un sistema capaz de detectar automáticamente reseñas negativas de películas. Para lograrlo, se utilizó un conjunto de datos de IMDb compuesto por miles de reseñas etiquetadas como positivas y negativas.

El proyecto consistió en construir y evaluar múltiples modelos de Procesamiento de Lenguaje Natural (NLP) para identificar correctamente el sentimiento de una reseña y seleccionar la solución con mejor desempeño.

---

## Objetivo de Negocio

Las plataformas de entretenimiento reciben diariamente grandes volúmenes de opiniones de usuarios. Analizar manualmente estas reseñas resulta costoso y poco escalable.

El objetivo de este proyecto fue desarrollar una solución automatizada capaz de:

- Identificar reseñas positivas y negativas.
- Analizar grandes volúmenes de texto en segundos.
- Detectar oportunidades de mejora en la experiencia del usuario.
- Automatizar procesos de monitoreo de opinión.
- Generar información útil para la toma de decisiones.

---

## ✅ Criterio de Éxito

El proyecto estableció como requisito principal:

```text
F1 Score ≥ 0.85
```

El modelo seleccionado debía alcanzar o superar este valor sobre datos no vistos.

---

## Descripción del Dataset

Se utilizó el conjunto de datos **IMDb Reviews**, compuesto por:

| Característica | Valor |
|---------------|--------|
| Total de reseñas | 47,331 |
| Variables | 17 |
| Tipo de problema | Clasificación binaria |
| Idioma | Inglés |
| Fuente | IMDb |

### Variable Objetivo

| Variable | Descripción |
|-----------|-------------|
| pos | 1 = Positiva, 0 = Negativa |

---

## Análisis Exploratorio de Datos (EDA)

### Distribución Temporal

Durante el análisis exploratorio se identificó que:

✅ La cantidad de películas aumenta considerablemente a partir de los años 70.

✅ La mayor concentración de títulos se encuentra entre finales de los años noventa y mediados de los años dos mil.

✅ El número de reseñas crece de forma importante en períodos recientes.

✅ Las reseñas positivas y negativas mantienen una distribución estable a lo largo del tiempo.

✅ Los registros posteriores a 2010 son escasos y deben interpretarse con cautela.

---

### Distribución de Reseñas por Película

El análisis de la cantidad de reseñas por película reveló que:

✅ La mayoría de películas poseen pocas reseñas.

✅ La distribución presenta una fuerte asimetría positiva.

✅ Existen películas con cantidades excepcionalmente altas de reseñas.

✅ La densidad se concentra en valores bajos.

Este comportamiento es común en plataformas donde pocos títulos concentran gran parte de las interacciones.

---

### Balance de Clases

| Clase | Cantidad |
|---------|---------:|
| Negativa (0) | 23,715 |
| Positiva (1) | 23,616 |

Diferencia:

```text
99 observaciones
```

Conclusión:

✅ Las clases se encuentran perfectamente balanceadas.

✅ No fue necesario aplicar técnicas de oversampling ni undersampling.

---

### Validación de Train y Test

Se verificó la distribución de puntuaciones y polaridades en ambos conjuntos.

Hallazgos:

✅ Distribuciones prácticamente idénticas.

✅ Sin evidencia de sesgo entre entrenamiento y prueba.

✅ Partición representativa de la población original.

✅ Base sólida para evaluación confiable de modelos.

---

## Tecnologías Utilizadas

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>

<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>

<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>

<img src="https://img.shields.io/badge/NLTK-154F73?style=for-the-badge"/>

<img src="https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge"/>

<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>

<img src="https://img.shields.io/badge/TF--IDF-2E7D32?style=for-the-badge"/>

<img src="https://img.shields.io/badge/LightGBM-FF9800?style=for-the-badge"/>

<img src="https://img.shields.io/badge/BERT-673AB7?style=for-the-badge"/>

<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>

</p>

---

## Preparación de los Datos

Para reducir el ruido y estandarizar las reseñas se aplicó una fase de normalización:

### Transformaciones realizadas

✅ Conversión de texto a minúsculas.

✅ Eliminación de números.

✅ Eliminación de signos de puntuación.

✅ Eliminación de caracteres especiales.

✅ Conservación únicamente de letras y espacios.

Ejemplo:

```text
"This Movie!!! 10/10"
↓
"this movie"
```

---

## Procesamiento de Lenguaje Natural

### NLTK

Se utilizaron listas de stopwords en inglés para eliminar palabras con bajo valor predictivo.

### spaCy

Se aplicó lematización para reducir la variabilidad del vocabulario.

Ejemplo:

```text
running
runs
ran

↓

run
```

Beneficios:

- Reducción de dimensionalidad.
- Mejor generalización.
- Consolidación de términos equivalentes.

---

## Vectorización de Texto

Las reseñas fueron transformadas mediante:

```text
TF-IDF (Term Frequency - Inverse Document Frequency)
```

Esta técnica permite representar cada documento mediante características numéricas que reflejan la importancia relativa de las palabras dentro del corpus.

---

## Estrategia de Evaluación

Todos los modelos fueron comparados utilizando:

- Accuracy
- F1 Score
- ROC-AUC
- Average Precision Score (APS)

Además, se analizaron:

- Curvas F1
- Curvas ROC
- Curvas Precision-Recall

---

# Modelos Evaluados

## Modelo Base

### DummyClassifier

Utilizado como línea base.

Resultados:

| Métrica | Test |
|----------|---------:|
| ROC-AUC | 0.50 |
| APS | 0.50 |

Conclusión:

Modelo equivalente al azar.

---

## Modelo 1

### TF-IDF + Logistic Regression

#### Entrenamiento

| Métrica | Valor |
|----------|---------:|
| Accuracy | 0.94 |
| F1 | 0.94 |
| APS | 0.98 |
| ROC-AUC | 0.98 |

#### Prueba

| Métrica | Valor |
|----------|---------:|
| Accuracy | 0.88 |
| F1 | 0.88 |
| APS | 0.95 |
| ROC-AUC | 0.95 |

✅ Superó ampliamente el objetivo del proyecto.

✅ Excelente capacidad de generalización.

✅ Mejor rendimiento global.

---

## Modelo 2

### spaCy + Lematización + TF-IDF + Logistic Regression

#### Prueba

| Métrica | Valor |
|----------|---------:|
| Accuracy | 0.87 |
| F1 | 0.87 |
| APS | 0.94 |
| ROC-AUC | 0.95 |

✅ Resultado competitivo.

✅ Ligera reducción respecto al modelo anterior.

---

## Modelo 3

### spaCy + TF-IDF + LightGBM

#### Prueba

| Métrica | Valor |
|----------|---------:|
| Accuracy | 0.85 |
| F1 | 0.85 |
| APS | 0.93 |
| ROC-AUC | 0.93 |

✅ Cumplió el requisito mínimo.

Rendimiento inferior a Regresión Logística.

---

## Modelo 4

### BERT + Logistic Regression

Se implementó una solución basada en:

```text
bert-base-uncased
```

El proceso incluyó:

✅ Tokenización.

✅ Embeddings contextualizados.

✅ Extracción de representaciones mediante el token [CLS].

✅ Clasificación utilizando Regresión Logística.

### Limitación Encontrada

```python
torch.cuda.is_available()
```

Resultado:

```text
False
```

El entorno no disponía de GPU, por lo que BERT fue ejecutado exclusivamente sobre CPU.

Debido al elevado costo computacional, se trabajó con una muestra reducida de datos.

---

# Comparación Final

| Modelo | F1 Test | ROC-AUC Test |
|----------|---------:|---------:|
| DummyClassifier | 0.50 | 0.50 |
| TF-IDF + Logistic Regression | **0.88** | **0.95** |
| spaCy + TF-IDF + Logistic Regression | 0.87 | 0.95 |
| spaCy + TF-IDF + LightGBM | 0.85 | 0.93 |

---

#  Modelo Seleccionado

## TF-IDF + Logistic Regression

Resultado final:

```text
F1 Score = 0.88
ROC-AUC = 0.95
```

Razones de selección:

✅ Mejor rendimiento global.

✅ Superó la meta de negocio.

✅ Excelente capacidad de generalización.

✅ Menor complejidad computacional.

✅ Más eficiente que modelos más sofisticados.

---

#  Prueba con Nuevas Reseñas

El modelo fue evaluado utilizando opiniones no presentes en el conjunto original.

| Reseña | Probabilidad Positiva |
|----------|---------:|
| This movie was absolutely amazing and the acting was fantastic. | 0.95 |
| One of the worst movies I have ever seen. | 0.46 |
| The story was interesting but the pacing was slow. | 0.19 |
| Excellent film. I would definitely watch it again. | 1.00 |
| Terrible plot and boring characters. | 0.00 |

### Resultado

✅ El modelo identifica correctamente opiniones claramente positivas.

✅ El modelo identifica correctamente opiniones claramente negativas.

✅ Presenta mayor incertidumbre en reseñas ambiguas o mixtas.

---

#  Conclusiones

✅ Se analizaron más de 47 mil reseñas reales de IMDb.

✅ El conjunto de datos se encontraba balanceado y adecuadamente particionado.

✅ Se exploraron enfoques clásicos y modernos de NLP.

✅ TF-IDF demostró ser una representación altamente efectiva para esta tarea.

✅ La Regresión Logística superó modelos más complejos como LightGBM.

✅ El modelo seleccionado alcanzó un F1 de 0.88 y un ROC-AUC de 0.95 sobre datos no vistos.

✅ Se superó exitosamente el criterio de éxito del proyecto (F1 ≥ 0.85).

✅ Las pruebas con reseñas personalizadas confirmaron la capacidad del modelo para identificar automáticamente el sentimiento predominante.

---

##  Impacto Empresarial

La solución desarrollada puede ser utilizada para:

- Monitoreo automático de opiniones de usuarios.
- Sistemas de recomendación.
- Estudios de satisfacción de clientes.
- Análisis masivo de comentarios.
- Moderación de contenido.
- Inteligencia de negocio basada en texto.

---

## Autor

### Santiago Rodríguez Zaraza

📊 Data Scientist en formación  
🤖 Machine Learning  
📝 Natural Language Processing (NLP)  
🐍 Python & SQL

---

<h3 align="center">
⭐ Si te gustó este proyecto, considera darle una estrella al repositorio.
</h3>
