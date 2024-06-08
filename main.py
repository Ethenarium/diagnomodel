import shutil
from datetime import datetime
import functools
import os
import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for, jsonify, current_app, Response, flash
from tensorflow import keras
from werkzeug.utils import secure_filename
import pymongo
from sklearn.metrics import f1_score
from decouple import config
import pandas as pd
from bson.objectid import ObjectId
import json
from flask_mail import Mail, Message
from bson.binary import Binary
import io
from tensorflow.keras.models import load_model
import time
import base64

app = Flask('app')  # Flask uygulaması oluşturur
app.secret_key = config('secret')  # Flask uygulamasının gizli anahtarını .env dosyasından ayarlar
my_client = pymongo.MongoClient(config('mongo_url'))  # MongoDB istemcisini .env dosyasındaki URL ile başlatır
my_db = my_client[config('db_name')]  # MongoDB veritabanını .env dosyasındaki isimle seçer
warning_models = []

model_to_dataset = {
        'active_models/cad.keras': '1.csv',
        'active_models/heart_attack.keras': '2.csv',
        'active_models/heart_failure.keras': '3.csv',
        'active_models/arrhythmias.keras': '4.csv',
        'active_models/hypertension.keras': '5.csv',
        'active_models/atherosclerosis.keras': '6.csv',
        'active_models/angina_pectoris.keras': '7.csv',
        'active_models/endocarditis.keras': '8.csv',
        'active_models/valvular_heart_disease.keras': '9.csv',
        'active_models/pericarditis.keras': '10.csv',
        'active_models/cardiomyopathy.keras': '11.csv',
        'active_models/congenital_heart_defects.keras': '12.csv',
        'active_models/pulmonary_embolism.keras': '13.csv',
        'active_models/cardiac_arrest.keras': '14.csv',
        'active_models/cardiogenic_shock.keras': '15.csv',
        'active_models/rheumatic_heart_disease.keras': '16.csv',
        'active_models/heart_valve_regurgitation.keras': '17.csv',
        'active_models/heart_valve_stenosis.keras': '18.csv',
        'active_models/heart_murmur.keras': '19.csv',
        'active_models/pad.keras': '20.csv'
    }


def evaluate_model_performance():
    for model_file in os.listdir('active_models'):
        model_name = model_file[:-6]
        dataset_filename = model_to_dataset.get('active_models/' + model_file)
        if dataset_filename:
            model_path = os.path.join('active_models', model_file)
            model = load_model(model_path)
            test_data = pd.read_csv('datasets/' + dataset_filename)
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            predictions = model.predict(X_test)
            predictions = np.argmax(predictions, axis=1)

            f1 = f1_score(y_test, predictions, average='macro')
            print(f"Processed {model_name}: F1-Score = {f1}")
            if f1 < 0.50:
                warning_models.append(model_name)


model_paths = {
    'angina_pectoris': 'active_models/angina_pectoris.keras',
    'arrhythmias': 'active_models/arrhythmias.keras',
    'atherosclerosis': 'active_models/atherosclerosis.keras',
    'cad': 'active_models/cad.keras',
    'cardiac_arrest': 'active_models/cardiac_arrest.keras',
    'cardiogenic_shock': 'active_models/cardiogenic_shock.keras',
    'cardiomyopathy': 'active_models/cardiomyopathy.keras',
    'congenital_heart_defects': 'active_models/congenital_heart_defects.keras',
    'endocarditis': 'active_models/endocarditis.keras',
    'heart_attack': 'active_models/heart_attack.keras',
    'heart_failure': 'active_models/heart_failure.keras',
    'heart_murmur': 'active_models/heart_murmur.keras',
    'heart_valve_regurgitation': 'active_models/heart_valve_regurgitation.keras',
    'heart_valve_stenosis': 'active_models/heart_valve_stenosis.keras',
    'hypertension': 'active_models/hypertension.keras',
    'pad': 'active_models/pad.keras',
    'pericarditis': 'active_models/pericarditis.keras',
    'pulmonary_embolism': 'active_models/pulmonary_embolism.keras',
    'rheumatic_heart_disease': 'active_models/rheumatic_heart_disease.keras',
    'valvular_heart_disease': 'active_models/valvular_heart_disease.keras'
}


loaded_models = {key: keras.models.load_model(path) for key, path in model_paths.items()}

@app.route('/activate_model/<model_id>', methods=['POST'])
def activate_model(model_id):
    model_record = my_db.models.find_one({"_id": ObjectId(model_id)})
    if not model_record:
        return jsonify({'error': 'Model not found'}), 404

    model_name = model_record['name']
    model_path = os.path.join('models', f"{model_name}.keras")

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found on server'}), 404

    try:
        active_model_path = os.path.join('active_models', f"{model_name}.keras")
        if os.path.exists(active_model_path):
            os.remove(active_model_path)
        shutil.copy(model_path, active_model_path)
        return jsonify({'message': 'Model activated successfully'}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to load model', 'message': str(e)}), 500


ALLOWED_EXTENSIONS = {'keras'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'modelFile' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['modelFile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        model_data = file.read()
        binary_model = Binary(model_data)
        model_document = {
            'name': request.form['diseaseSelect'],
            'model_data': binary_model,
            'upload_date': datetime.now()
        }
        my_db.models.insert_one(model_document)
        flash('Model successfully uploaded')
        return redirect(url_for('modelhub'))
    else:
        flash('Invalid file type')
        return redirect(request.url)


@app.route('/download/<model_id>', methods=['GET'])
def download(model_id):
    model_record = my_db.models.find_one({"_id": ObjectId(model_id)})
    if not model_record:
        flash("Model not found.")
        return redirect(url_for('modelhub'))

    binary_model = model_record['model_data']
    model_name = model_record['name']
    model_path = os.path.join('models', f"{model_name}.keras")

    if not os.path.exists('models'):
        os.makedirs('models')
    with open(model_path, 'wb') as f:
        f.write(binary_model)

    return jsonify({'message': 'Model saved successfully', 'model_path': model_path}), 200


@app.route('/api/models', methods=['GET'])
def list_models():
    try:
        models = my_db.models.find({}, {'name': 1, 'upload_date': 1})
        models_list = [{'_id': str(model['_id']), 'name': model['name'], 'upload_date': model['upload_date']} for model in models]
        return jsonify(models_list)
    except Exception as e:
        return jsonify({'error': 'Failed to fetch models', 'message': str(e)}), 500


def get_sequence(seq_name):
    return my_db.counters.find_one_and_update(filter={"_id": seq_name}, update={"$inc": {"seq": 1}}, upsert=True)["seq"]


def get_patient_data(patient_id):
    patient_record = my_db.patientData.find_one({"_id": ObjectId(patient_id)})
    return patient_record


def prepare_patient_data(patient_data):

    feature_names = [
        "Age", "BMI", "Smoker", "Drinker", "Headache", "Fatigue", "ChestPain",
        "ShortnessOfBreath", "Dizziness", "Palpitations", "Swelling", "IrregularHeartbeat",
        "Nausea", "ColdSweats", "Indigestion", "PainArm", "JawPain", "BackPain",
        "Fainting", "PersistentCough", "DifficultySleeping", "SuddenWeightGain",
        "RapidPulse", "ExtremeWeakness", "LossOfConsciousness", "BlurredVision",
        "LegPain", "NumbnessInLimbs", "ColdLimbs", "Cyanosis", "PoorGrowthInInfants",
        "RecurrentRespiratoryInfections", "Sweating", "Confusion", "NumbnessOrWeaknessInLegs",
        "ColdnessInLowerLegOrFoot", "SoresOrWoundsOnToesFeetOrLegs"
    ]

    features = [
        patient_data['age'],
        patient_data['bmi'],
        patient_data['symptoms'].get('smoker', 0),
        patient_data['symptoms'].get('drinker', 0),
        patient_data['symptoms'].get('headache', 0),
        patient_data['symptoms'].get('chestPain', 0),
        patient_data['symptoms'].get('shortnessOfBreath', 0.5),
        patient_data['symptoms'].get('fatigue', 0),
        patient_data['symptoms'].get('fever', 0),
        patient_data['symptoms'].get('dizziness', 0),
        patient_data['symptoms'].get('swelling', 0),
        patient_data['symptoms'].get('irregularHeartbeat', 0),
        patient_data['symptoms'].get('nauseaOrVomit', 0),
        patient_data['symptoms'].get('coldSweats', 0),
        patient_data['symptoms'].get('indigestionOrHeartburn', 0),
        patient_data['symptoms'].get('painToArm', 0),
        patient_data['symptoms'].get('jawPain', 0),
        patient_data['symptoms'].get('backPain', 0),
        patient_data['symptoms'].get('fainting', 0),
        patient_data['symptoms'].get('persistentCough', 0),
        patient_data['symptoms'].get('difficultySleeping', 0),
        patient_data['symptoms'].get('suddenWeightGain', 0),
        patient_data['symptoms'].get('rapidOrIrregularPulse', 0),
        patient_data['symptoms'].get('extremeWeakness', 0),
        patient_data['symptoms'].get('lossOfConsciousness', 0),
        patient_data['symptoms'].get('blurredVision', 0),
        patient_data['symptoms'].get('legPain', 0),
        patient_data['symptoms'].get('numbnessInLimbs', 0),
        patient_data['symptoms'].get('coldLimbs', 0),
        patient_data['symptoms'].get('cyanosis', 0),
        patient_data['symptoms'].get('poorGrowthInInfants', 0),
        patient_data['symptoms'].get('recurrentRespiratoryInfections', 0),
        patient_data['symptoms'].get('sweating', 0),
        patient_data['symptoms'].get('confusion', 0),
        patient_data['symptoms'].get('numbnessOrWeaknessInLegs', 0),
        patient_data['symptoms'].get('coldnessInLowerLegOrFoot', 0),
        patient_data['symptoms'].get('soresOrWoundsOnLegsOrFeet', 0)
    ]

    features_df = pd.DataFrame([features], columns=feature_names)
    return features_df


def my_log(action, message, user_name):
    log_id = get_sequence("log")
    return my_db.logs.insert_one({
        "_id": log_id,
        "action": action,
        "message": message,
        "user_name": user_name,
        "log_date": datetime.now()
    })


def my_logon(username, password):
    user_info = my_db.users.find_one({"_id": username})
    if user_info and password == user_info.get("password"):
        session["user_info"] = user_info
        return user_info
    else:
        return None


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


def login_required(route):
    @functools.wraps(route)
    def route_wrapper(*args, **kwargs):
        if session.get("user_info") is None:
            return redirect(url_for("login"))
        return route(*args, **kwargs)
    return route_wrapper


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        username = request.form.get("username")
        pwd = request.form.get("pwd")
        user_info = my_logon(username=username, password=pwd)
        if user_info:
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for("login"))
    return render_template("login.html")


@app.route('/dashboard')
def dashboard():
    total_patients = my_db.patientData.count_documents({})

    gender_distribution = list(my_db.patientData.aggregate([
        {"$group": {"_id": "$gender", "count": {"$sum": 1}}}
    ]))

    age_distribution = list(my_db.patientData.aggregate([
        {"$bucket": {
            "groupBy": "$age",
            "boundaries": [0, 18, 30, 45, 60, 75, 100],
            "default": "100+",
            "output": {
                "count": {"$sum": 1}
            }
        }}
    ]))

    recent_patients_cursor = my_db.patientData.find({}, {'name': 1, 'diagnosis': 1}).sort("date_diagnosed", -1).limit(5)
    recent_patients = [patient for patient in recent_patients_cursor]

    pipeline = [
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date_added"}}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    patient_counts = list(my_db.patientData.aggregate(pipeline))

    model_performance = {}
    for model_file in os.listdir('active_models'):
        model_name = model_file[:-6]
        dataset_filename = model_to_dataset.get('active_models/' + model_file)
        if dataset_filename:
            model_path = os.path.join('active_models', model_file)
            model = load_model(model_path)
            test_data = pd.read_csv('datasets/' + dataset_filename)
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            predictions = model.predict(X_test)
            predictions = np.argmax(predictions, axis=1)

            f1 = f1_score(y_test, predictions, average='macro')
            model_performance[model_name] = {'f1_score': f1}
            print(f"Processed {model_name}: F1-Score = {f1}")
        else:
            print(f"Dataset not found for {model_name}")

    diagnosis_frequency = list(my_db.patientData.aggregate([
        {"$group": {"_id": "$diagnosis", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]))

    return render_template('dashboard.html', total_patients=total_patients,
                           recent_patients=recent_patients,
                           patient_counts=patient_counts,
                           model_performance=model_performance,
                           gender_distribution=gender_distribution,
                           age_distribution=age_distribution,
                           diagnosis_frequency=diagnosis_frequency)


@app.route('/patient_details/<string:patient_id>')
@login_required
def patient_details(patient_id):
    patient = my_db.patientData.find_one({"_id": ObjectId(patient_id)})
    if patient:
        return render_template('patient_details.html', patient=patient)
    else:
        return 'Patient not found', 404


@app.route('/dataTable')
def data_table():
    patients = list(my_db.patientData.find())
    return render_template('dataTable.html', patients=patients)


@app.route('/')
@login_required
def index():
    return render_template('main.html')


@app.route('/modelhub')
@login_required
def modelhub():
    return render_template('ModelHub.html')


@app.route('/submit', methods=['POST'])
@login_required
def submit():
    name = request.form.get("name")
    email = request.form.get("email")
    age = int(request.form.get("age", 0))
    weight = float(request.form.get("weight", 0))
    height = float(request.form.get("height", 0))
    bloodPressureMin = float(request.form.get("bloodPressureMin", 0))
    bloodPressureMax = float(request.form.get("bloodPressureMax", 0))

    bmi = weight / ((height / 100) ** 2) if height > 0 else 0

    patient_data = {
        "_id": ObjectId(),
        "name": name,
        "email": email,
        "gender": request.form.get("gender", "None"),
        "age": age,
        "weight": weight,
        "height": height,
        "bmi": round(bmi, 2),
        "familyHistory": request.form.get("familyHistory", "None"),
        "bloodPressureMin": bloodPressureMin,
        "bloodPressureMax": bloodPressureMax,
        "date_added": datetime.now(),
        "blood_values": [],
        "symptoms": [

        ]
    }

    symptoms_data = json.loads(request.form.get('symptoms', '{}'))
    patient_data["symptoms"] = symptoms_data

    if 'bloodValues' in request.files:
        file = request.files['bloodValues']
        if file and file.filename.endswith('.xlsx'):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            df = pd.read_excel(filepath)
            blood_values = df.to_dict(orient='records')

            patient_data["blood_values"].extend(blood_values)

            os.remove(filepath)

    if 'patientImage' in request.files:
        image_file = request.files['patientImage']
        if image_file.filename != '':
            filename = secure_filename(image_file.filename)
            content_type = image_file.content_type
            image_bytes = io.BytesIO(image_file.read())
            binary_image = Binary(image_bytes.getvalue())
            patient_data['image'] = binary_image
            patient_data['image_type'] = content_type

    my_db.patientData.insert_one(patient_data)

    return redirect(url_for('index'))


@app.route('/get_image/<patient_id>')
def get_image(patient_id):
    patient = my_db.patientData.find_one({"_id": ObjectId(patient_id)})
    if patient and 'image' in patient:
        return Response(patient['image'], mimetype=patient.get('image_type', 'image/png'))
    return 'No image found', 404


@app.template_filter('symptom_text')
def symptom_text(value):
    if value == 0:
        return 'Never'
    elif value == 0.25:
        return 'Low'
    elif value == 0.50:
        return 'Medium'
    elif value == 1:
        return 'High'
    else:
        return 'Unknown'


@app.template_filter('symptom_class')
def symptom_class(value):
    if value == 0.25:
        return 'text-success'
    elif value == 0.50:
        return 'text-warning'
    elif value == 1:
        return 'text-danger'
    else:
        return ''


@app.route('/api/diagnosis', methods=['GET'])
def get_all_inventory():
    try:
        patients = list(my_db.patientData.find({}, projection={'image': False}))
        for patient in patients:
            patient['_id'] = str(patient['_id'])
        return jsonify(patients)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/doctor/<email>', methods=['GET'])
@login_required
def get_doctor_by_email(email):
    try:
        doctor = my_db.users.find_one({"_id": email}, {"_id": 0, "full_name": 1})
        if doctor:
            return jsonify(doctor)
        else:
            return jsonify({'error': 'Doctor not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        return super(JSONEncoder, self).default(obj)


app.json_encoder = JSONEncoder


@app.route('/api/patients/<string:patient_name>', methods=['GET'])
@login_required
def get_patient_by_name(patient_name):
    try:
        patient = my_db.patientData.find_one({"name": patient_name})
        if patient:
            return jsonify(patient)
        else:
            return jsonify({'error': 'Patient not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def classify_symptoms(symptoms):
    diseases = {
        'No Diagnosis': [],
        'Cad': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'IrregularHeartbeat', 'ExtremeWeakness', 'BlurredVision'],
        'Heart Attack': ['ChestPain', 'ShortnessOfBreath', 'Nausea', 'ColdSweats', 'PainArm', 'PoorGrowthInInfants'],
        'Heart Failure': ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'PersistentCough', 'RapidPulse', 'JawPain'],
        'Arrhythmias': ['Palpitations', 'Dizziness', 'Fainting', 'RapidPulse', 'Fatigue', 'ChestPain'],
        'Hypertension': ['Headache', 'Dizziness', 'BlurredVision', 'ShortnessOfBreath', 'Cyanosis', 'ColdSweats'],
        'Atherosclerosis': ['ChestPain', 'LegPain', 'NumbnessInLimbs', 'ColdLimbs', 'ExtremeWeakness', 'Confusion'],
        'Angina Pectoris': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Nausea', 'IrregularHeartbeat', 'Cyanosis'],
        'Endocarditis': ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever', 'ColdSweats'],
        'Valvular Heart Disease': ['ShortnessOfBreath', 'ChestPain', 'Fatigue', 'Dizziness', 'Swelling', 'Drinker'],
        'Pericarditis': ['ChestPain', 'Fever', 'ShortnessOfBreath', 'Fatigue', 'Drinker', 'RapidPulse'],
        'Cardiomyopathy': ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'IrregularHeartbeat', 'Dizziness','Indigestion'],
        'Congenital Heart Defects': ['Cyanosis', 'ShortnessOfBreath', 'PoorGrowthInInfants', 'Fatigue','RecurrentRespiratoryInfections', 'Nausea'],
        'Pulmonary Embolism': ['ShortnessOfBreath', 'ChestPain', 'RapidPulse', 'Sweating', 'Indigestion', 'Fever'],
        'Cardiac Arrest': ['LossOfConsciousness', 'Fainting', 'ExtremeWeakness', 'Nausea', 'BackPain', 'LegPain'],
        'Cardiogenic Shock': ['RapidPulse', 'Fatigue', 'ColdSweats', 'Confusion', 'DifficultySleeping','RecurrentRespiratoryInfections'],
        'Rheumatic Heart Disease': ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever', 'Sweating','IrregularHeartbeat'],
        'Heart Valve Regurgitation': ['Fatigue', 'ShortnessOfBreath', 'Palpitations', 'Swelling', 'ColdnessInLowerLegOrFoot', 'Cyanosis'],
        'Heart Valve Stenosis': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Fainting', 'SoresOrWoundsOnToesFeetOrLegs', 'Nausea'],
        'Heart Murmur': ['ShortnessOfBreath', 'Fatigue', 'Dizziness', 'Swelling', 'Confusion', 'Drinker'],
        'Peripheral Artery Disease (PAD)': ['LegPain', 'NumbnessOrWeaknessInLegs', 'ColdnessInLowerLegOrFoot', 'SoresOrWoundsOnToesFeetOrLegs', 'Drinker']
    }

    best_match = 'No Diagnosis'
    max_count = 0
    max_intensity = 0

    for disease, relevant_symptoms in diseases.items():
        symptom_count = sum(symptom in symptoms for symptom in relevant_symptoms)
        symptom_intensity = sum(symptoms.get(symptom, 0) for symptom in relevant_symptoms if symptom in symptoms)

        if symptom_count > max_count or (symptom_count == max_count and symptom_intensity > max_intensity):
            model_name = f"{disease.lower().replace(' ', '_')}.keras"
            if model_name not in warning_models:
                best_match = disease
                max_count = symptom_count
                max_intensity = symptom_intensity

    return best_match


def process_input(symptoms):
    disease = classify_symptoms(symptoms)
    if disease:
        model_path = f"active_models/{disease.lower().replace(' ', '_')}.keras"
        if model_path not in warning_models:
            model = load_model(model_path)
            input_data = np.array([list(symptoms.values())])
            prediction = model.predict(input_data)
            return disease, prediction[0]
    return 'No disease detected or insufficient data', None



@app.route('/diagnose/<string:patient_id>', methods=['POST'])
def diagnose_patient(patient_id):
    try:
        current_app.logger.info(f"Diagnosing patient with ID: {patient_id}")
        patient_data = get_patient_data(patient_id)
        if not patient_data:
            return jsonify({'error': 'Patient not found'}), 404

        symptoms = {
            'Age': patient_data.get('age',0),
            'BMI': patient_data.get('bmi',0),
            'Smoker': patient_data['symptoms'].get('smoke', 0),
            'Drinker': patient_data['symptoms'].get('drinker', 0),
            'Headache': patient_data['symptoms'].get('headache', 0),
            'Fatigue': patient_data['symptoms'].get('fatigue', 0),
            'Fever': patient_data['symptoms'].get('fever',0),
            'ChestPain': patient_data['symptoms'].get('chestPain', 0),
            'ShortnessOfBreath': patient_data['symptoms'].get('shortnessOfBreath', 0),
            'Dizziness': patient_data['symptoms'].get('dizziness', 0),
            'Palpitations': patient_data['symptoms'].get('palpitations', 0),
            'Swelling': patient_data['symptoms'].get('swelling', 0),
            'IrregularHeartbeat': patient_data['symptoms'].get('irregularHeartbeat', 0),
            'Nausea': patient_data['symptoms'].get('nauseaOrVomit', 0),
            'ColdSweats': patient_data['symptoms'].get('coldSweats', 0),
            'Indigestion': patient_data['symptoms'].get('indigestionOrHeartburn', 0),
            'PainArm': patient_data['symptoms'].get('painToArm', 0),
            'JawPain': patient_data['symptoms'].get('jawPain', 0),
            'BackPain': patient_data['symptoms'].get('backPain', 0),
            'Fainting': patient_data['symptoms'].get('fainting', 0),
            'PersistentCough': patient_data['symptoms'].get('persistentCough', 0),
            'DifficultySleeping': patient_data['symptoms'].get('difficultySleeping', 0),
            'SuddenWeightGain': patient_data['symptoms'].get('suddenWeightGain', 0),
            'RapidPulse': patient_data['symptoms'].get('rapidOrIrregularPulse', 0),
            'ExtremeWeakness': patient_data['symptoms'].get('extremeWeakness', 0),
            'LossOfConsciousness': patient_data['symptoms'].get('lossOfConsciousness', 0),
            'BlurredVision': patient_data['symptoms'].get('blurredVision', 0),
            'LegPain': patient_data['symptoms'].get('legPain', 0),
            'NumbnessInLimbs': patient_data['symptoms'].get('numbnessInLimbs', 0),
            'ColdLimbs': patient_data['symptoms'].get('coldLimbs', 0),
            'Cyanosis': patient_data['symptoms'].get('cyanosis', 0),
            'PoorGrowthInInfants': patient_data['symptoms'].get('poorGrowthInInfants', 0),
            'RecurrentRespiratoryInfections': patient_data['symptoms'].get('recurrentRespiratoryInfections', 0),
            'Sweating': patient_data['symptoms'].get('sweating', 0),
            'Confusion': patient_data['symptoms'].get('confusion', 0),
            'NumbnessOrWeaknessInLegs': patient_data['symptoms'].get('numbnessOrWeaknessInLegs', 0),
            'ColdnessInLowerLegOrFoot': patient_data['symptoms'].get('coldnessInLowerLegOrFoot', 0),
            'SoresOrWoundsOnToesFeetOrLegs': patient_data['symptoms'].get('soresOrWoundsOnLegsOrFeet', 0),
        }

        disease, prediction = process_input(symptoms)
        if disease in warning_models:
            return jsonify(
                {'error': f"The model for {disease} is below reliability threshold. Immediate update required."}), 422

        if disease == 'No disease detected or insufficient data':
            return jsonify({'error': 'No disease detected or insufficient data'}), 422

        my_db.patientData.update_one({"_id": ObjectId(patient_id)}, {"$set": {"diagnosis": disease}})
        send_diagnosis_email(patient_data.get('email'))
        return jsonify({'diagnosis': disease}), 200

    except Exception as e:
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500


app.config.update(
    MAIL_SERVER='smtp.office365.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='erberk38@outlook.com',
    MAIL_PASSWORD='Erberk7268bankai+Seftali+',
    MAIL_DEFAULT_SENDER='erberk38@outlook.com'
)

mail = Mail(app)


def send_diagnosis_email(email):
    msg = Message("Teşhis Sonuçlarınız",
                  recipients=[email])
    msg.body = (
        "Sayın Hasta,\n\n"
        "Teşhis sonuçlarınız çıkmıştır. Lütfen en kısa sürede hastanemize gerekli tedavi için başvurunuz.\n\n"
        "Sağlığınız bizim için önemlidir. Herhangi bir sorunuz veya endişeniz varsa, bizimle iletişime geçmekten çekinmeyin.\n\n"
        "Saygılarımızla,\n"
        "Vital Verse Hastanesi"
    )

    retries = 3
    for attempt in range(retries):
        try:
            mail.send(msg)
            print("Email sent successfully")
            return
        except Exception as e:
            wait = 2 ** attempt
            print(f"Failed to send email, retrying in {wait} seconds...")
            time.sleep(wait)
    print("Failed to send email after several attempts.")


if __name__ == '__main__':
    app.run(debug=True)