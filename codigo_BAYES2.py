from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# -------------------------------
# Datos simulados (los mismos de la imagen)
# -------------------------------
data = {
    'text': [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños"
    ],
    'label': [
        'real', 'fake', 'real', 'fake', 'real',
        'fake', 'real', 'fake', 'real', 'fake'
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# Preprocesamiento y modelo
# -------------------------------
X = df['text']
y = df['label']

# Convertir texto en vectores de frecuencia
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Entrenar el clasificador Naive Bayes con TODOS los datos
model = MultinomialNB()
model.fit(X_vectorized, y)

# -------------------------------
# Evaluación del modelo (sobre los mismos datos de entrenamiento)
# -------------------------------
y_pred = model.predict(X_vectorized)

print("--- REPORTE DEL MODELO ---")
print("Reporte de clasificación:")
print(classification_report(y, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y, y_pred, labels=['real', 'fake']))

# -------------------------------
# Clasificación de nuevas noticias
# -------------------------------
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

nuevas_vectorizadas = vectorizer.transform(nuevas_noticias)
predicciones = model.predict(nuevas_vectorizadas)

print("\n--- CLASIFICACIÓN DE NUEVOS MENSAJES ---")
for noticia, prediccion in zip(nuevas_noticias, predicciones):
    print(f'Noticia: "{noticia}" → Clasificación: {prediccion}')