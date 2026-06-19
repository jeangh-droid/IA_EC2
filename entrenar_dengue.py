# -*- coding: utf-8 -*-
"""
Script de Entrenamiento Local para el Modelo de Dengue
Este script se ejecuta directamente en tu entorno virtual local (VSC)
donde ya tienes instaladas las versiones correspondientes:
TensorFlow 2.10, Pandas 2.2.3, Numpy 2.2.4 y Scikit-Learn.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. CONFIGURACIÓN DE RUTAS
# El script se ejecutará desde la raíz del proyecto (IA_EC2)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'airweb', 'datos_abiertos_vigilancia_dengue_2000_2024.csv') # Ajusta el nombre si tu CSV se llama diferente

print("=== INICIANDO ENTRENAMIENTO LOCAL ===")

if not os.path.exists(DATA_PATH):
    # Intentar buscar cualquier archivo .csv en la raíz o subcarpetas si no encuentra el por defecto
    print(f"Advertencia: No se encontró el archivo en {DATA_PATH}")
    print("Buscando un archivo CSV alternativo en el directorio actual...")
    csv_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.csv')]
    if csv_files:
        DATA_PATH = os.path.join(BASE_DIR, csv_files[0])
        print(f"Se utilizará el archivo encontrado: {DATA_PATH}")
    else:
        print("ERROR: Por favor coloca tu archivo de datos (CSV) en la raíz del proyecto o dentro de airweb/")
        exit()

# 2. CARGA Y LIMPIEZA DE DATOS (Compatible con Pandas 2.2.3 y Numpy 2.2.4)
df = pd.read_csv(DATA_PATH, sep=';', on_bad_lines='skip')

print("\n=== VALORES INICIALES DE ENFERMEDAD ===")
print(df['enfermedad'].value_counts())

# Corregir codificación BOM si existe en la columna departamento
df = df.rename(columns={'ï»¿departamento': 'departamento'})

# Filtrar únicamente las columnas necesarias
columnas_usar = ['departamento', 'ano', 'semana', 'edad', 'tipo_edad', 'sexo', 'diresa', 'enfermedad']
df = df[columnas_usar].copy()

# Eliminar nulos de forma segura
df = df.dropna()

# Convertir columna edad a numérico evitando conflictos de tipos
df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
df = df.dropna(subset=['edad'])

# Estandarizar textos a mayúsculas limpias
for col in ['departamento', 'tipo_edad', 'sexo', 'enfermedad']:
    df[col] = df[col].astype(str).str.strip().str.upper()

print(f"\nDimensiones del dataset procesado: {df.shape}")

# 3. PREPARACIÓN DE CARACTERÍSTICAS (ENCODING & SCALING)
cat_cols = ['departamento', 'tipo_edad', 'sexo']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Guardar mapeos de variables categóricas
joblib.dump(encoders, os.path.join(BASE_DIR, 'encoder_dengue.pkl'))
print("-> encoder_dengue.pkl guardado ✓")

# Separar variables predictoras (X) y objetivo (y)
X = df.drop('enfermedad', axis=1)
y_raw = df['enfermedad']

# Codificar variable objetivo
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)
clases = le_target.classes_.tolist()

# Guardar metadatos estructurales requeridos por Django
with open(os.path.join(BASE_DIR, 'clases_dengue.json'), 'w') as f:
    json.dump(clases, f)

with open(os.path.join(BASE_DIR, 'columnas_dengue.json'), 'w') as f:
    json.dump(X.columns.tolist(), f)

# Escalar/Estandarizar variables numéricas de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler_dengue.pkl'))
print("-> scaler_dengue.pkl, clases_dengue.json y columnas_dengue.json guardados ✓")

# División de conjuntos de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. ENTRENAMIENTO DE RED NEURONAL (Nativo en TensorFlow 2.10.0)
num_clases = len(clases)
y_train_cat = to_categorical(y_train, num_clases)
y_test_cat  = to_categorical(y_test,  num_clases)

modelo = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_clases, activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nEntrenando Red Neuronal en entorno local...")
modelo.fit(
    X_train, y_train_cat,
    epochs=20,
    batch_size=512,
    validation_split=0.1,
    verbose=1
)

loss, acc = modelo.evaluate(X_test, y_test_cat, verbose=0)
print(f"Accuracy final de la Red Neuronal: {acc:.4f}")

# Al estar usando TensorFlow 2.10 local, el guardado en formato .h5 es nativo
# y perfectamente compatible con el motor de carga de tu views.py.
modelo.save(os.path.join(BASE_DIR, 'modelo_dengue.h5'))
print("-> modelo_dengue.h5 guardado con firma HDF5 clásica ✓")

# 5. ENTRENAMIENTO DEL ÁRBOL DE DECISIONES
print("\nEntrenando Árbol de Decisiones...")
arbol = DecisionTreeClassifier(max_depth=10, random_state=42)
arbol.fit(X_train, y_train)

y_pred = arbol.predict(X_test)
print("\n=== REPORTE DE CLASIFICACIÓN (ÁRBOL) ===")
print(classification_report(y_test, y_pred, target_names=clases))

joblib.dump(arbol, os.path.join(BASE_DIR, 'arbol_dengue.pkl'))
print("-> arbol_dengue.pkl guardado ✓")

print("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")
print("Todos los artefactos generados se encuentran en la raíz de tu proyecto.")
print("Ya puedes arrancar tu servidor Django usando: python airweb/manage.py runserver")