import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template, Blueprint # Añadido Blueprint
from flask_restx import Api, Resource, fields, reqparse
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

# --- Descargar recursos de NLTK ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    WordNetLemmatizer().lemmatize("test")
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Cargar modelo y binarizador
MODEL_FILENAME = 'Genero_peliculas_pipeline.joblib'
MLB_FILENAME = 'mlb.joblib'

pipeline_model = None
mlb = None
stop_words_set = None
lemmatizer_g = None

try:
    pipeline_model = joblib.load(MODEL_FILENAME)
    mlb = joblib.load(MLB_FILENAME)
    stop_words_set = set(stopwords.words('english'))
    lemmatizer_g = WordNetLemmatizer()
    print("Modelo, binarizador y herramientas de limpieza cargados exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró '{MODEL_FILENAME}' o '{MLB_FILENAME}'.")
except Exception as e:
    print(f"Error al cargar modelo/binarizador o herramientas de limpieza: {e}")

# Configurar Flask App
app = Flask(__name__)

# --- Configurar Blueprint para la API con Flask-RESTX ---
# Esto ayuda a organizar y prefijar todas las rutas de la API.
api_bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')

api = Api(
    api_bp, # Inicializar Api con el Blueprint
    version='1.0',
    title='API de Predicción de Géneros de Películas',
    description='API que predice los géneros más probables a partir de la sinopsis de una película.',
    doc='/docs/' # La documentación de Swagger estará en /api/v1/docs/
)

# Registrar el Blueprint en la aplicación Flask principal
app.register_blueprint(api_bp)

# --- Namespace para la API RESTful (ahora relativo a /api/v1) ---
# El endpoint completo será /api/v1/predict/
ns_api = api.namespace('predict', description='API para predicción programática')

parser_api = reqparse.RequestParser()
parser_api.add_argument('sinopsis', type=str, required=True, help='Sinopsis de la película', location='args')

genre_prediction_fields_api = api.model('GenrePredictionAPI', {
    'predicted_genres': fields.List(fields.String),
    'predicted_probabilities_for_top_genres': fields.List(fields.Float),
    'all_genre_probabilities': fields.Raw
})

def clean_text_api(text):
    if not isinstance(text, str) or lemmatizer_g is None or stop_words_set is None:
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer_g.lemmatize(word) for word in words if word not in stop_words_set and len(word) > 1]
    return " ".join(words)

def predict_genres_logic(sinopsis_text):
    """Lógica de predicción separada para ser usada por la API y la UI."""
    if pipeline_model is None or mlb is None:
        print("Error: Modelo o binarizador no cargados en predict_genres_logic.")
        return -1, [], [], {}

    try:
        cleaned_plot_text = clean_text_api(sinopsis_text)
        
        if not cleaned_plot_text:
            print("Advertencia: La sinopsis resultó vacía después de la limpieza.")
            # Corrección: Usar mlb.classes_ para generar nombres de columna correctos
            empty_probs = {'p_' + g: 0.0 for g in mlb.classes_}
            return 1, [], [], empty_probs

        all_proba = pipeline_model.predict_proba([cleaned_plot_text])[0]
        all_genre_probabilities_dict = {'p_' + mlb.classes_[i]: float(all_proba[i]) for i in range(len(mlb.classes_))}

        N_TOP_GENRES = 3
        top_n_indices = np.argsort(all_proba)[-N_TOP_GENRES:][::-1]
        
        predicted_genres = mlb.classes_[top_n_indices].tolist()
        predicted_probs_for_top_genres = all_proba[top_n_indices].tolist()
        
        return 1, predicted_genres, predicted_probs_for_top_genres, all_genre_probabilities_dict

    except Exception as e:
        print(f"Error durante la predicción en la API: {e}")
        import traceback
        traceback.print_exc()
        return -2, [], [], {}

@ns_api.route('/') # Este endpoint ahora es /api/v1/predict/
class GenrePredictionApi(Resource):
    @ns_api.expect(parser_api)
    def get(self):
        args = parser_api.parse_args()
        status, genres, top_probs, all_probs_dict = predict_genres_logic(args['sinopsis'])
        if status < 0:
            api.abort(500, "Error interno del servidor durante la predicción.")
        return {
            'predicted_genres': genres,
            'predicted_probabilities_for_top_genres': top_probs,
            'all_genre_probabilities': all_probs_dict
        }, 200

# --- Ruta para la Interfaz Web (UI) en la raíz de la aplicación ---
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_genres = []
    predicted_probabilities = []
    all_genre_probabilities = {}
    error_message = None
    submitted = False 

    if request.method == 'POST':
        submitted = True
        sinopsis_text = request.form.get('sinopsis_text', '')
        
        if not sinopsis_text.strip():
            error_message = "Por favor, ingresa una sinopsis."
        elif pipeline_model is None or mlb is None:
            error_message = "Error del servidor: El modelo no está cargado."
        else:
            status, genres, probs, all_probs_dict = predict_genres_logic(sinopsis_text)
            if status > 0:
                predicted_genres = genres
                predicted_probabilities = probs
                all_genre_probabilities = all_probs_dict
            elif status == -1:
                error_message = "Error del servidor: El modelo no está disponible."
            else:
                error_message = "Ocurrió un error durante la predicción."
                
    return render_template('index.html', 
                           predicted_genres=predicted_genres, 
                           predicted_probabilities=predicted_probabilities,
                           all_genre_probabilities=all_genre_probabilities,
                           error_message=error_message,
                           submitted=submitted)

# Ejecutar app
if __name__ == '__main__':
    if pipeline_model is None or mlb is None:
        print("No se pudo iniciar la API/UI porque el modelo o el binarizador no están cargados.")
    else:
        print("Iniciando servidor Flask para UI y API...")
        print("Tu UI estará en: http://localhost:5000/")
        print("La documentación de tu API RESTful estará en: http://localhost:5000/api/v1/docs/")
        print("El endpoint de la API RESTful estará en: http://localhost:5000/api/v1/predict/?sinopsis=...")
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

