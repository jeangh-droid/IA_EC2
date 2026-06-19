import os
import json
import joblib
import numpy as np
import tensorflow as tf
from django.shortcuts import render

# Ruta base correspondiente a airweb/predictor/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# ── CARGA DE MODELOS Y COMPONENTES ────────
# ==========================================

# ── TITANIC (Modelo Local TF 2.10) ───────────────────────────
model_path_titanic = os.path.join(BASE_DIR, '..', 'modelo_accidente.keras')
model = tf.keras.models.load_model(model_path_titanic)


# ── DENGUE (Carga Nativa HDF5 del Modelo Completo Local) ─────
# Al haber entrenado en tu propia PC, cargamos el archivo .h5 directamente
modelo_dengue = tf.keras.models.load_model(os.path.join(BASE_DIR, '..', 'modelo_dengue.h5'))

# Cargar archivos de configuración estructural (JSON)
with open(os.path.join(BASE_DIR, '..', 'columnas_dengue.json')) as f:
    columnas_dengue = json.load(f)

with open(os.path.join(BASE_DIR, '..', 'clases_dengue.json')) as f:
    clases_dengue = json.load(f)

# Cargar los componentes de Scikit-Learn usando Joblib
arbol_dengue    = joblib.load(os.path.join(BASE_DIR, '..', 'arbol_dengue.pkl'))
scaler_dengue   = joblib.load(os.path.join(BASE_DIR, '..', 'scaler_dengue.pkl'))
encoders_dengue = joblib.load(os.path.join(BASE_DIR, '..', 'encoder_dengue.pkl'))


# ==========================================
# ── VISTAS DE DJANGO ──────────────────────
# ==========================================

# ── VISTA TITANIC ─────────────────────────────────────────
def index(request):
    prediction = None
    percentage = None
    error      = None

    if request.method == 'POST':
        try:
            sex          = request.POST.get('sex')
            age          = float(request.POST.get('age'))
            siblings     = float(request.POST.get('siblings'))
            parch        = float(request.POST.get('parch'))
            fare         = float(request.POST.get('fare'))
            travel_class = request.POST.get('travel_class')
            deck         = request.POST.get('deck')
            embark_town  = request.POST.get('embark_town')
            alone        = request.POST.get('alone')

            data = {
                'sex':                np.array([sex]),
                'age':                np.array([age],                dtype=np.float32),
                'n_siblings_spouses': np.array([siblings],           dtype=np.float32),
                'parch':              np.array([parch],              dtype=np.float32),
                'fare':               np.array([fare],               dtype=np.float32),
                'class':              np.array([travel_class]),
                'deck':               np.array([deck]),
                'embark_town':        np.array([embark_town]),
                'alone':              np.array([alone]),
            }

            result      = model(data)
            probability = float(tf.sigmoid(result)[0][0])
            percentage  = round(probability * 100, 2)
            prediction  = "Sobrevive" if probability >= 0.5 else "No sobrevive"

        except Exception as e:
            error = str(e)

    return render(request, 'predictor/index.html', {
        'prediction': prediction,
        'percentage': percentage,
        'error':      error,
    })


# ── VISTA DENGUE ──────────────────────────────────────────
DEPARTAMENTOS = [
    'AMAZONAS','ANCASH','APURIMAC','AREQUIPA','AYACUCHO','CAJAMARCA',
    'CALLAO','CUSCO','HUANCAVELICA','HUANUCO','ICA','JUNIN',
    'LA LIBERTAD','LAMBAYEQUE','LIMA','LORETO','MADRE DE DIOS',
    'MOQUEGUA','PASCO','PIURA','PUNO','SAN MARTIN','TACNA',
    'TUMBES','UCAYALI',
]

def dengue(request):
    resultado_red   = None
    probabilidad    = None
    resultado_arbol = None
    error           = None

    if request.method == 'POST':
        try:
            departamento = request.POST.get('departamento', '').strip().upper()
            ano          = int(request.POST.get('ano', 2024))
            semana       = int(request.POST.get('semana', 1))
            edad         = float(request.POST.get('edad', 0))
            tipo_edad    = request.POST.get('tipo_edad', 'A').strip().upper()
            sexo         = request.POST.get('sexo', 'M').strip().upper()
            diresa       = float(request.POST.get('diresa', 16))

            dep_enc  = encoders_dengue['departamento'].transform([departamento])[0]
            tipo_enc = encoders_dengue['tipo_edad'].transform([tipo_edad])[0]
            sexo_enc = encoders_dengue['sexo'].transform([sexo])[0]

            entrada        = np.array([[dep_enc, ano, semana, edad, tipo_enc, sexo_enc, diresa]])
            entrada_scaled = scaler_dengue.transform(entrada)

            # Red Neuronal (Predicción)
            pred_red      = modelo_dengue.predict(entrada_scaled)
            idx_red       = int(np.argmax(pred_red))
            resultado_red = clases_dengue[idx_red]
            probabilidad  = round(float(pred_red[0][idx_red]) * 100, 2)

            # Árbol de Decisiones (Predicción)
            pred_arbol      = arbol_dengue.predict(entrada_scaled)
            resultado_arbol = clases_dengue[int(pred_arbol[0])]

        except Exception as e:
            error = str(e)

    return render(request, 'predictor/dengue.html', {
        'resultado_red'  : resultado_red,
        'probabilidad'   : probabilidad,
        'resultado_arbol': resultado_arbol,
        'error'          : error,
        'departamentos'  : DEPARTAMENTOS,
    })