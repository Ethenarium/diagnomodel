from datetime import datetime, timedelta
import functools
import os
from flask import Flask, redirect, render_template, request, session, url_for, jsonify
from werkzeug.utils import secure_filename  # secure_filename için eklendi
import pymongo
from decouple import config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # pandas için eklendi
from bson.objectid import ObjectId


app = Flask('app')  # Flask uygulaması oluşturur
app.secret_key = config('secret')  # Flask uygulamasının gizli anahtarını .env dosyasından ayarlar
my_client = pymongo.MongoClient(config('mongo_url'))  # MongoDB istemcisini .env dosyasındaki URL ile başlatır
my_db = my_client[config('db_name')]  # MongoDB veritabanını .env dosyasındaki isimle seçer

if not os.path.exists('uploads'):  # Eğer 'uploads' klasörü mevcut değilse,
    os.makedirs('uploads')  # 'uploads' klasörünü oluşturur


def get_sequence(seq_name):  # Veritabanında belirli bir sayaç için sıra numarası alır veya oluşturur
    return my_db.counters.find_one_and_update(filter={"_id": seq_name}, update={"$inc": {"seq": 1}}, upsert=True)["seq"]


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


@app.route('/submit', methods=['POST'])
@login_required
def submit():
    # Form verilerini al
    name = request.form.get("name")
    age = int(request.form.get("age"))
    weight = float(request.form.get("weight", 0))  # Eğer değer yoksa 0 olarak kabul et
    height = float(request.form.get("height", 0))  # Eğer değer yoksa 0 olarak kabul et

    # BMI hesaplaması
    bmi = weight / ((height / 100) ** 2) if height > 0 else 0

    # Semptomları formdan al ve listeye dönüştür
    symptoms = request.form.getlist("symptoms")

    patient_data = {
        "_id": ObjectId(),
        "name": name,
        "age": age,
        "weight": weight,
        "height": height,
        "bmi": round(bmi, 2),  # BMI değerini yuvarla
        "familyHistory": request.form.get("familyHistory"),
        "bloodPressure": request.form.get("bloodPressure"),
        "symptoms": symptoms,  # Semptom listesini ekle
        "blood_values": []
    }

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


if __name__ == '__main__':
    app.run(debug=True)