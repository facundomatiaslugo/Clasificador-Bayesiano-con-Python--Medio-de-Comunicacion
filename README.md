# Clasificador Bayesiano para Detección de Noticias Falsas (Fake News)

**Alumno:** Facundo Lugo  
**Materia:** Procesamiento de Aprendizaje Automático  
**Profesora:** Yanina Scudero  
**Carrera:** Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial  

---

## 📌 Contexto

Un medio de comunicación necesita implementar un sistema automático para detectar si una noticia es **falsa (`fake`)** o **verdadera (`real`)**, basándose únicamente en su contenido textual.

## 🎯 Objetivo

Entrenar un modelo de **Naive Bayes** utilizando un conjunto de noticias etiquetadas, evaluar su rendimiento y utilizarlo para clasificar nuevas noticias que el sistema no haya visto antes.

---

## 🛠️ Tecnologías Utilizadas

*   **Python 3**
*   **Pandas:** Para la manipulación y estructuración de los datos.
*   **Scikit-learn:** 
    *   `CountVectorizer` (para convertir texto a vectores numéricos).
    *   `MultinomialNB` (algoritmo Naive Bayes para clasificación de texto).
    *   `metrics` (para el reporte de clasificación y matriz de confusión).

---

## 📂 Estructura del Repositorio

*   `README.md`: Este archivo explicativo.
*   `codigo_BAYES2.py`: Script principal con la solución del ejercicio.

---

## 🧠 Explicación del Código (`codigo_BAYES2.py`)

El script realiza los siguientes pasos:

1.  **Carga de Datos:** Se define un DataFrame con 10 noticias (5 reales y 5 falsas) extraídas del enunciado.
2.  **Preprocesamiento:** Se utiliza `CountVectorizer` para transformar el texto de las noticias en una matriz de frecuencia de palabras (Bag of Words). Esto permite que el algoritmo procese lenguaje humano.
3.  **Entrenamiento:** Se entrena un modelo `MultinomialNB` con el 100% de los datos para que aprenda los patrones asociados a cada clase (`real` / `fake`).
4.  **Evaluación:** Se predicen las etiquetas sobre el mismo conjunto de datos para verificar la capacidad de aprendizaje del modelo.
5.  **Predicción:** Se introducen dos noticias nuevas (no vistas en el entrenamiento) y el modelo devuelve su clasificación.

---

## 📊 Resultados Obtenidos

Al ejecutar el script, se obtiene el siguiente informe por consola:

### 1. Reporte de Clasificación (Métricas)

              precision    recall  f1-score   support

        fake       1.00      1.00      1.00         5
        real       1.00      1.00      1.00         5

    accuracy                           1.00        10
   macro avg       1.00      1.00      1.00        10
weighted avg       1.00      1.00      1.00        10

**Interpretación:**
*   **Precisión (Accuracy): 100% (1.00).** El modelo clasificó correctamente las 10 noticias del conjunto de datos.
*   **Precision, Recall y F1-Score:** Todos los valores son 1.00, lo que indica un rendimiento perfecto sobre los datos de entrenamiento. Esto es esperable dado el tamaño reducido del dataset y que se evaluó sobre los mismos datos con los que se entrenó.

### 2. Matriz de Confusión

[[5 0]
 [0 5]]

**Interpretación:**

|                | Predijo: Fake | Predijo: Real |
| :------------- | :-----------: | :-----------: |
| **Era: Fake**  |       5       |       0       |
| **Era: Real**  |       0       |       5       |

*   **5 (Verdaderos Fake):** Noticias falsas correctamente identificadas.
*   **0 (Falsos Real):** No hubo noticias falsas clasificadas erróneamente como reales.
*   **0 (Falsos Fake):** No hubo noticias reales clasificadas erróneamente como falsas.
*   **5 (Verdaderos Real):** Noticias reales correctamente identificadas.
*   **Conclusión:** El modelo no cometió ningún error. Todos los valores fuera de la diagonal son 0.

### 3. Clasificación de Nuevas Noticias

--- CLASIFICACIÓN DE NUEVOS MENSAJES ---
Noticia: "Nuevo estudio demuestra que el café mejora la memoria" → Clasificación: fake
Noticia: "Expertos afirman que los gatos pueden hablar con humanos" → Clasificación: fake

**Interpretación:**
*   El modelo predijo que **ambas noticias nuevas son falsas (`fake`)**.
*   **¿Por qué?** 
    *   La primera noticia contiene la palabra **"estudio"**, la cual en el entrenamiento apareció asociada a noticias falsas (ej: *"Estudio revela que comer chocolate cura el cáncer"*).
    *   La segunda noticia contiene la frase **"Expertos afirman"**, muy similar a **"Científicos afirman"** (que también era `fake` en el entrenamiento). Además, la afirmación sobre gatos hablando es absurda, y el modelo aprendió que ese tipo de titulares suelen ser falsos.

---

## ✅ Conclusión

El modelo **Naive Bayes** logró aprender satisfactoriamente los patrones lingüísticos que diferencian a las noticias reales de las falsas en el conjunto de datos proporcionado. 

Para los nuevos mensajes, el modelo determinó que ambos son **falsos**, basándose en la similitud de su vocabulario con las noticias falsas del conjunto de entrenamiento. Esto demuestra la utilidad de los clasificadores bayesianos para tareas de Procesamiento de Lenguaje Natural (PLN), especialmente cuando se cuenta con datos etiquetados.

**Limitación:** El dataset es muy pequeño (10 noticias). Para un sistema de producción real, se requeriría un conjunto de datos mucho más grande y variado para evitar el sobreajuste (overfitting) y mejorar la capacidad de generalización.

---

## 🚀 Cómo Ejecutar

1.  Asegúrate de tener Python instalado.
2.  Instala las dependencias necesarias:
    pip install pandas scikit-learn
3.  Ejecuta el script:
    python codigo_BAYES2.py
4.  Observa los resultados en la consola.

---

**Facundo Lugo**  
*Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial*
