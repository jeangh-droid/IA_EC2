from django.shortcuts import render
import tensorflow as tf
import numpy as np
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model_path = os.path.join(
    BASE_DIR,
    '..',
    'modelo_accidente.keras'
)

model = tf.keras.models.load_model(model_path)


def index(request):

    prediction = None
    percentage = None
    error = None

    if request.method == 'POST':

        try:

            sex = request.POST.get('sex')
            age = float(request.POST.get('age'))
            siblings = float(request.POST.get('siblings'))
            parch = float(request.POST.get('parch'))
            fare = float(request.POST.get('fare'))
            travel_class = request.POST.get('travel_class')
            deck = request.POST.get('deck')
            embark_town = request.POST.get('embark_town')
            alone = request.POST.get('alone')

            data = {

                'sex': np.array([sex]),

                'age': np.array([age], dtype=np.float32),

                'n_siblings_spouses': np.array(
                    [siblings],
                    dtype=np.float32
                ),

                'parch': np.array(
                    [parch],
                    dtype=np.float32
                ),

                'fare': np.array(
                    [fare],
                    dtype=np.float32
                ),

                'class': np.array([travel_class]),

                'deck': np.array([deck]),

                'embark_town': np.array([embark_town]),

                'alone': np.array([alone])

            }

            result = model(data)

            probability = float(
                tf.sigmoid(result)[0][0]
            )

            percentage = round(
                probability * 100,
                2
            )

            if probability >= 0.5:
                prediction = "Sobrevive"
            else:
                prediction = "No sobrevive"

        except Exception as e:

            error = str(e)

    return render(
        request,
        'predictor/index.html',
        {
            'prediction': prediction,
            'percentage': percentage,
            'error': error
        }
    )