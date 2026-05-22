# -*- coding: utf-8 -*-

"""
Proyecto: Predicción de supervivencia
Modelo TensorFlow + Django
Versión corregida y compatible con Python 3.10
"""

# =========================
# IMPORTACIONES
# =========================

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Normalization,
    StringLookup,
    CategoryEncoding
)

# =========================
# CONFIGURACIÓN NUMPY
# =========================

np.set_printoptions(precision=3, suppress=True)

# =========================
# LEER DATASET
# =========================

air_accident = pd.read_csv("train.csv")

print("\nPrimeros registros:")
print(air_accident.head())

# =========================
# SEPARAR FEATURES Y LABELS
# =========================

air_accident_features = air_accident.copy()

# Columna objetivo
air_accident_labels = air_accident_features.pop('survived')

# =========================
# ENTRADAS SIMBÓLICAS
# =========================

inputs = {}

for name, column in air_accident_features.items():

    dtype = column.dtype

    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(
        shape=(1,),
        name=name,
        dtype=dtype
    )

print("\nInputs creados:")
print(inputs)

# =========================
# PREPROCESAMIENTO NUMÉRICO
# =========================

numeric_inputs = {
    name: input
    for name, input in inputs.items()
    if input.dtype == tf.float32
}

x = layers.Concatenate()(list(numeric_inputs.values()))

norm = Normalization()

norm.adapt(
    np.array(
        air_accident[numeric_inputs.keys()]
    )
)

all_numeric_inputs = norm(x)

# =========================
# PREPROCESAMIENTO CATEGÓRICO
# =========================

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():

    if input.dtype == tf.float32:
        continue

    lookup = StringLookup(
        vocabulary=np.unique(
            air_accident_features[name]
        )
    )

    one_hot = CategoryEncoding(
        num_tokens=lookup.vocabulary_size()
    )

    x = lookup(input)

    x = one_hot(x)

    preprocessed_inputs.append(x)

# =========================
# CONCATENAR ENTRADAS
# =========================

preprocessed_inputs_cat = layers.Concatenate()(
    preprocessed_inputs
)

# =========================
# MODELO DE PREPROCESAMIENTO
# =========================

air_accident_preprocessing = tf.keras.Model(
    inputs,
    preprocessed_inputs_cat
)

# =========================
# GRAFICAR MODELO
# =========================

tf.keras.utils.plot_model(
    model=air_accident_preprocessing,
    rankdir="LR",
    dpi=72,
    show_shapes=True,
    to_file="modelo.png"
)

print("\nModelo gráfico guardado como modelo.png")

# =========================
# CONVERTIR A DICCIONARIO
# =========================

air_accident_features_dict = {
    name: np.array(value)
    for name, value in air_accident_features.items()
}

# =========================
# PROBAR PREPROCESAMIENTO
# =========================

features_dict = {
    name: values[:1]
    for name, values in air_accident_features_dict.items()
}

print("\nSalida del preprocesamiento:")
print(
    air_accident_preprocessing(features_dict)
)

# =========================
# CONSTRUIR MODELO
# =========================

def build_model(preprocessing_head, inputs):

    body = tf.keras.Sequential([

        layers.Dense(
            64,
            activation='relu'
        ),

        layers.Dense(
            32,
            activation='relu'
        ),

        layers.Dense(1)

    ])

    preprocessed_inputs = preprocessing_head(inputs)

    result = body(preprocessed_inputs)

    model = tf.keras.Model(inputs, result)

    model.compile(
        loss=tf.losses.BinaryCrossentropy(
            from_logits=True
        ),
        optimizer=tf.optimizers.Adam(),
        metrics=['accuracy']
    )

    return model

# =========================
# CREAR MODELO
# =========================

air_accident_model = build_model(
    air_accident_preprocessing,
    inputs
)

# =========================
# RESUMEN DEL MODELO
# =========================

print("\nResumen del modelo:")
air_accident_model.summary()

# =========================
# ENTRENAMIENTO
# =========================

print("\nEntrenando modelo...")

history = air_accident_model.fit(
    x=air_accident_features_dict,
    y=air_accident_labels,
    epochs=10,
    validation_split=0.2
)

# =========================
# GUARDAR MODELO
# =========================

air_accident_model.save(
    "modelo_accidente.keras"
)

print("\nModelo guardado correctamente.")

# =========================
# RECARGAR MODELO
# =========================

reloaded = tf.keras.models.load_model(
    "modelo_accidente.keras"
)

# =========================
# VALIDAR MODELO
# =========================

before = air_accident_model(features_dict)

after = reloaded(features_dict)

assert np.all(
    np.abs(before - after) < 1e-3
)

print("\nPredicción original:")
print(before)

print("\nPredicción recargada:")
print(after)

# =========================
# ITERADOR PERSONALIZADO
# =========================

def slices(features):

    for i in itertools.count():

        example = {
            name: values[i]
            for name, values in features.items()
        }

        yield example

# =========================
# MOSTRAR PRIMER EJEMPLO
# =========================

print("\nPrimer ejemplo:")

for example in slices(
    air_accident_features_dict
):

    for name, value in example.items():

        print(f"{name:19s}: {value}")

    break

# =========================
# TF.DATA.DATASET
# =========================

features_ds = tf.data.Dataset.from_tensor_slices(
    air_accident_features_dict
)

print("\nEjemplo desde tf.data.Dataset:")

for example in features_ds:

    for name, value in example.items():

        print(f"{name:19s}: {value}")

    break

# =========================
# DATASET COMPLETO
# =========================

air_accident_ds = tf.data.Dataset.from_tensor_slices(

    (
        air_accident_features_dict,
        air_accident_labels
    )

)

air_accident_batches = air_accident_ds.shuffle(
    len(air_accident_labels)
).batch(32)

# =========================
# ENTRENAMIENTO CON DATASET
# =========================

print("\nEntrenamiento usando tf.data.Dataset...")

air_accident_model.fit(
    air_accident_batches,
    epochs=5
)

# =========================
# LEER CSV COMO DATASET
# =========================

air_accident_file_path = "train.csv"

air_accident_csv_ds = tf.data.experimental.make_csv_dataset(

    air_accident_file_path,

    batch_size=5,

    label_name='survived',

    num_epochs=1,

    ignore_errors=True,

)

# =========================
# MOSTRAR BATCH CSV
# =========================

print("\nBatch desde CSV Dataset:")

for batch, label in air_accident_csv_ds.take(1):

    for key, value in batch.items():

        print(f"{key:20s}: {value}")

    print()

    print(f"{'label':20s}: {label}")

print("\nProceso finalizado correctamente.")