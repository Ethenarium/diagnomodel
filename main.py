from datetime import datetime
import functools
import os
from flask import Flask, redirect, render_template, request, session, url_for, jsonify, current_app
from werkzeug.utils import secure_filename
import pymongo
from decouple import config
import pandas as pd
from bson.objectid import ObjectId
import json
import joblib
from flask_mail import Mail, Message

app = Flask('app')  # Flask uygulaması oluşturur
app.secret_key = config('secret')  # Flask uygulamasının gizli anahtarını .env dosyasından ayarlar
my_client = pymongo.MongoClient(config('mongo_url'))  # MongoDB istemcisini .env dosyasındaki URL ile başlatır
my_db = my_client[config('db_name')]  # MongoDB veritabanını .env dosyasındaki isimle seçer
model = joblib.load('logistic_regression_model.pkl')

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

    # Veritabanına patientData ekleyin
    my_db.patientData.insert_one(patient_data)

    return redirect(url_for('index'))


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
@login_required
def get_all_inventory():
    try:
        patients = list(my_db.patientData.find())
        for patient in patients:
            patient['_id'] = str(patient['_id'])
        return jsonify(patients)
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


app.config.update(
    MAIL_SERVER='smtp.office365.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='anakinskyw0@hotmail.com',
    MAIL_PASSWORD='Seftali6872++',
    MAIL_DEFAULT_SENDER='anakinskyw0@hotmail.com'
)

mail = Mail(app)


@app.route('/diagnose/<string:patient_id>', methods=['POST'])
def diagnose_patient(patient_id):
    max_retries = 100
    try:
        current_app.logger.info(f"Diagnosing patient with ID: {patient_id}")

        patient_data = get_patient_data(patient_id)
        if not patient_data:
            return jsonify({'error': 'Patient not found'}), 404

        diagnosis_name = "Unknown Diagnosis"
        for attempt in range(max_retries):
            if diagnosis_name == "Unknown Diagnosis":
                if attempt > 0:
                    current_app.logger.info(f"Retrying diagnosis for patient ID: {patient_id}, attempt: {attempt + 1}")

                prepared_data = prepare_patient_data(patient_data)
                diagnosis_id = model.predict(prepared_data)
                current_app.logger.info(f"Model predicted diagnosis ID: {diagnosis_id}")
                diagnosis_name = get_diagnosis_name(diagnosis_id[0])

                if diagnosis_name != "Unknown Diagnosis":
                    break
            else:
                break

        if diagnosis_name == "Unknown Diagnosis":
            diagnosis_name = "Failed Diagnosis"

        current_app.logger.info(f"Diagnosis name retrieved: {diagnosis_name}")

        result = my_db.patientData.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": {"diagnosis": diagnosis_name}}
        )
        if result.modified_count == 1:
            send_diagnosis_email(patient_data.get('email', ''), diagnosis_name)
            current_app.logger.info(f"Database updated and email sent for patient ID: {patient_id}")
            return jsonify({'diagnosis': diagnosis_name})
        else:
            raise Exception("Failed to update the patient's diagnosis in the database.")
    except Exception as e:
        current_app.logger.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500


def send_diagnosis_email(email, diagnosis):
    msg = Message("Your Diagnosis Result",
                  recipients=[email])
    msg.body = f"Dear Patient,\n\nYour diagnosis result is: {diagnosis}.\nPlease consult with your physician for further advice.\n\nBest regards,\nMedical Team"
    mail.send(msg)


if __name__ == '__main__':
    app.run(debug=True)