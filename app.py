import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template # Añadir render_template
from flask_restx import Api, Resource, fields, reqparse
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os # Para os.path.join si es necesario

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
MODEL_FILENAME = 'Genero_peliculas_pipeline.joblib' # Asegúrate que este sea tu pipeline simple (solo TF-IDF)
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

# Configurar Flask
app = Flask(__name__) # Flask buscará la carpeta 'templates' aquí
api = Api(
    app,
    version='1.0',
    title='API y UI de Predicción de Géneros de Películas',
    description='Predice los géneros más probables a partir de la sinopsis de una película.'
)

# --- Namespace para la API RESTful (la que ya tenías) ---
ns_api = api.namespace('predict_api', description='API para predicción programática')

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
            empty_probs = {'p_' + g.replace(' ', '_').replace('-', '_'): 0.0 for g in mlb.classes_} # Corrección aquí
            return 1, [], [], empty_probs

        # El pipeline simple (TF-IDF -> Classifier) espera un iterable de strings
        all_proba = pipeline_model.predict_proba([cleaned_plot_text])[0]

        all_genre_probabilities_dict = {'p_' + g: float(all_proba[i]) for i, g in enumerate(mlb.classes_)}

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

@ns_api.route('/')
class GenrePredictionApi(Resource):
    @ns_api.expect(parser_api)
    # @ns_api.marshal_with(genre_prediction_fields_api) # Puedes ajustar y descomentar
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

# --- Nueva Ruta para la Interfaz Web (UI) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_genres = []
    predicted_probabilities = []
    all_genre_probabilities = {}
    error_message = None
    submitted = False # Para saber si el formulario fue enviado

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
            elif status == -1: # Error de modelo no cargado
                error_message = "Error del servidor: El modelo no está disponible."
            else: # Otro error de predicción
                error_message = "Ocurrió un error durante la predicción."
                
    # Pasa las variables a la plantilla
    return render_template('index.html', 
                           predicted_genres=predicted_genres, 
                           predicted_probabilities=predicted_probabilities,
                           all_genre_probabilities=all_genre_probabilities,
                           error_message=error_message,
                           submitted=submitted) # Para saber si mostrar mensaje de "no se predijeron"

# Ejecutar app
if __name__ == '__main__':
    if pipeline_model is None or mlb is None:
        print("No se pudo iniciar la API/UI porque el modelo o el binarizador no están cargados.")
    else:
        print("Iniciando servidor Flask para UI y API...")
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

