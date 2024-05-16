from datetime import datetime
import functools
import os
import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for, jsonify, current_app, Response
from werkzeug.utils import secure_filename
import pymongo
from decouple import config
import pandas as pd
from bson.objectid import ObjectId
import json
import joblib
from flask_mail import Mail, Message
from bson.binary import Binary
import io
import base64
from tensorflow.keras.models import load_model
import time

app = Flask('app')  # Flask uygulaması oluşturur
app.secret_key = config('secret')  # Flask uygulamasının gizli anahtarını .env dosyasından ayarlar
my_client = pymongo.MongoClient(config('mongo_url'))  # MongoDB istemcisini .env dosyasındaki URL ile başlatır
my_db = my_client[config('db_name')]  # MongoDB veritabanını .env dosyasındaki isimle seçer

angina_pectoris_model = load_model('angina_pectoris_model.keras')
arrhythmias_model = load_model('arrhythmias_model.keras')
atherosclerosis_model = load_model('atherosclerosis_model.keras')
cad_model = load_model('cad_model.keras')
cardiac_arrest_model = load_model('cardiac_arrest_model.keras')
cardiogenic_shock_model = load_model('cardiogenic_shock_model.keras')
cardiomyopathy_model = load_model('cardiomyopathy_model.keras')
congenital_heart_defects_model = load_model('congenital_heart_defects_model.keras')
endocarditis_model = load_model('endocarditis_model.keras')
heart_attack_model = load_model('heart_attack_model.keras')
heart_failure_model = load_model('heart_failure_model.keras')
heart_murmur_model = load_model('heart_murmur_model.keras')
heart_valve_regurgitation_model = load_model('heart_valve_regurgitation_model.keras')
heart_valve_stenosis_model = load_model('heart_valve_stenosis_model.keras')
hypertension_model = load_model('hypertension_model.keras')
pad_model = load_model('pad_model.keras')
pericarditis_model = load_model('pericarditis_model.keras')
pulmonary_embolism_model = load_model('pulmonary_embolism_model.keras')
rheumatic_heart_disease_model = load_model('rheumatic_heart_disease_model.keras')
valvular_heart_disease_model = load_model('valvular_heart_disease_model.keras')

if not os.path.exists('uploads'):  # Eğer 'uploads' klasörü mevcut değilse,
    os.makedirs('uploads')  # 'uploads' klasörünü oluşturur


def get_sequence(seq_name):  # Veritabanında belirli bir sayaç için sıra numarası alır veya oluşturur
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


def get_diagnosis_name(diagnosis_id):
    diagnosis_map = {
        1: "CAD",
        2: "Heart Attack",
        3: "Heart Failure",
        4: "Arrhythmias",
        5: "Hypertension",
        6: "Atherosclerosis",
        7: "Angina Pectoris",
        8: "Endocarditis",
        9: "Valvular Heart Disease",
        10: "Pericarditis",
        11: "Cardiomyopathy",
        12: "Congenital Heart Defects",
        13: "Pulmonary Embolism",
        14: "Cardiac Arrest",
        15: "Cardiogenic Shock",
        16: "Rheumatic Heart Disease",
        17: "Heart Valve Regurgitation",
        18: "Heart Valve Stenosis",
        19: "Heart Murmur",
        20: "Peripheral Artery Disease (PAD)"
    }
    return diagnosis_map.get(diagnosis_id, "Unknown Diagnosis")


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
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route('/dataTable')
def data_table():
    patients = list(my_db.patientData.find())
    return render_template('dataTable.html', patients=patients)


@app.route('/')
@login_required
def index():
    return render_template('main.html')


@app.route('/ref')
@login_required
def ref():
    return render_template('ref.html')


@app.route('/submit', methods=['POST'])
@login_required
def submit():
    # Form verilerini al
    name = request.form.get("name")
    email = request.form.get("email")
    age = int(request.form.get("age", 0))  # Use 0 as default if age is not provided
    weight = float(request.form.get("weight", 0))  # Eğer değer yoksa 0 olarak kabul et
    height = float(request.form.get("height", 0))  # Eğer değer yoksa 0 olarak kabul et
    bloodPressureMin = float(request.form.get("bloodPressureMin", 0))
    bloodPressureMax = float(request.form.get("bloodPressureMax", 0))

    # BMI hesaplaması
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
        "familyHistory": request.form.get("familyHistory", "None"),  # Default value if not provided
        "bloodPressureMin": bloodPressureMin,
        "bloodPressureMax": bloodPressureMax,
        "blood_values": [],
        "symptoms": [

        ]
    }

    symptoms_data = json.loads(request.form.get('symptoms', '{}'))
    # Semptomları patient_data içine ekle
    patient_data["symptoms"] = symptoms_data

    # Kan değerlerini Excel dosyasından oku
    if 'bloodValues' in request.files:
        file = request.files['bloodValues']
        if file and file.filename.endswith('.xlsx'):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            # Excel dosyasını oku
            df = pd.read_excel(filepath)
            blood_values = df.to_dict(orient='records')  # DataFrame'i dictionary listesine dönüştür

            # Okunan kan değerlerini patient_data içine ekleyin
            patient_data["blood_values"].extend(blood_values)

            os.remove(filepath)  # Dosyayı işledikten sonra sil

    if 'patientImage' in request.files:
        image_file = request.files['patientImage']
        if image_file.filename != '':
            filename = secure_filename(image_file.filename)
            content_type = image_file.content_type
            image_bytes = io.BytesIO(image_file.read())
            binary_image = Binary(image_bytes.getvalue())
            patient_data['image'] = binary_image
            patient_data['image_type'] = content_type

    # Veritabanına patientData ekleyin
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
        # Use projection to exclude the 'image' field
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


@app.route('/api/patients/<string:patient_name>', methods=['GET'])
@login_required
def get_patient_by_name(patient_name):
    try:
        patient = my_db.patientData.find_one({"name": patient_name})
        if patient:
            patient['_id'] = str(patient['_id'])
            return jsonify(patient)
        else:
            return jsonify({'error': 'Patient not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def classify_symptoms(symptoms):
    diseases = {
        'Cad': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'IrregularHeartbeat', 'ExtremeWeakness', 'BlurredVision'], # CAD
        'Heart Attack': ['ChestPain', 'ShortnessOfBreath', 'Nausea', 'ColdSweats', 'PainArm', 'PoorGrowthInInfants'], # Heart Attack
        'Heart Failure': ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'PersistentCough', 'RapidPulse', 'JawPain'], # Heart Failure
        'Arrhythmias': ['Palpitations', 'Dizziness', 'Fainting', 'RapidPulse', 'Fatigue', 'ChestPain'],  # Arrhythmias
        'Hypertension': ['Headache', 'Dizziness', 'BlurredVision', 'ShortnessOfBreath', 'Cyanosis', 'ColdSweats'],  # Hypertension
        'Atherosclerosis': ['ChestPain', 'LegPain', 'NumbnessInLimbs', 'ColdLimbs', 'ExtremeWeakness', 'Confusion'],  # Atherosclerosis
        'Angina Pectoris': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Nausea', 'IrregularHeartbeat', 'Cyanosis'],  # Angina Pectoris
        'Endocarditis': ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever', 'ColdSweats'],  # Endocarditis
        'Valvular Heart Disease': ['ShortnessOfBreath', 'ChestPain', 'Fatigue', 'Dizziness', 'Swelling', 'Drinker'],  # Valvular Heart Disease
        'Pericarditis': ['ChestPain', 'Fever', 'ShortnessOfBreath', 'Fatigue', 'Drinker', 'RapidPulse'],  # Pericarditis
        'Cardiomyopathy': ['ShortnessOfBreath', 'Fatigue', 'Swelling', 'IrregularHeartbeat', 'Dizziness','Indigestion'],  # Cardiomyopathy
        'Congenital Heart Defects': ['Cyanosis', 'ShortnessOfBreath', 'PoorGrowthInInfants', 'Fatigue','RecurrentRespiratoryInfections', 'Nausea'],  # Congenital Heart Defects
        'Pulmonary Embolism': ['ShortnessOfBreath', 'ChestPain', 'RapidPulse', 'Sweating', 'Indigestion', 'Fever'],  # Pulmonary Embolism
        'Cardiac Arrest': ['LossOfConsciousness', 'Fainting', 'ExtremeWeakness', 'Nausea', 'BackPain', 'LegPain'],  # Cardiac Arrest
        'Cardiogenic Shock': ['RapidPulse', 'Fatigue', 'ColdSweats', 'Confusion', 'DifficultySleeping','RecurrentRespiratoryInfections'],  # Cardiogenic Shock
        'Rheumatic Heart Disease': ['Fatigue', 'ShortnessOfBreath', 'ChestPain', 'Fever', 'Sweating','IrregularHeartbeat'],  # Rheumatic Heart Disease
        'Heart Valve Regurgitation': ['Fatigue', 'ShortnessOfBreath', 'Palpitations', 'Swelling', 'ColdnessInLowerLegOrFoot', 'Cyanosis'],  # Heart Valve Regurgitation
        'Heart Valve Stenosis': ['ChestPain', 'ShortnessOfBreath', 'Fatigue', 'Fainting', 'SoresOrWoundsOnToesFeetOrLegs', 'Nausea'],  # Heart Valve Stenosis
        'Heart Murmur': ['ShortnessOfBreath', 'Fatigue', 'Dizziness', 'Swelling', 'Confusion', 'Drinker'], # Heart Murmur
        'Peripheral Artery Disease (PAD)': ['LegPain', 'NumbnessOrWeaknessInLegs', 'ColdnessInLowerLegOrFoot', 'SoresOrWoundsOnToesFeetOrLegs', 'Drinker'] # Peripheral Artery Disease (PAD)
    }

    best_match = None
    max_count = 0
    max_intensity = 0

    for disease, relevant_symptoms in diseases.items():
        symptom_count = sum(symptom in symptoms for symptom in relevant_symptoms)
        symptom_intensity = sum(symptoms[symptom] for symptom in relevant_symptoms if symptom in symptoms)

        if symptom_count > max_count or (symptom_count == max_count and symptom_intensity > max_intensity):
            best_match = disease
            max_count = symptom_count
            max_intensity = symptom_intensity

    return best_match


def process_input(symptoms):
    disease = classify_symptoms(symptoms)
    if disease:
        model = load_model(f"{disease.lower().replace(' ', '_')}_model.keras")
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
        if disease == 'No disease detected or insufficient data':
            return jsonify({'error': 'No disease detected or insufficient data'}), 422

        current_app.logger.info(f"Disease diagnosed: {disease}")

        # Update the database
        result = my_db.patientData.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": {"diagnosis": disease}}
        )

        send_diagnosis_email("testmailaras@gmail.com", disease)
        return jsonify({'diagnosis': disease}), 200

    except Exception as e:
        current_app.logger.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500


app.config.update(
    MAIL_SERVER='smtp.office365.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='anakinskyw0@hotmail.com',
    MAIL_PASSWORD='Seftali6872++',
    MAIL_DEFAULT_SENDER='anakinskyw0@hotmail.com'
)

mail = Mail(app)


def send_diagnosis_email(email, diagnosis):
    msg = Message("Your Diagnosis Result",
                  recipients=[email])
    msg.body = (f"Dear Patient,\n\nYour diagnosis result is: {diagnosis}."
                f"\nPlease come back for the check as soon as possible.\n"
                f"\nPlease consult with your physician for further advice.\n\nBest regards,\nMedical Team")

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