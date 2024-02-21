from datetime import datetime, timedelta
import functools
import os
from flask import Flask, redirect, render_template, request, session, url_for, jsonify, send_file
from werkzeug.utils import secure_filename  # secure_filename için eklendi
import pymongo
from decouple import config
import pandas as pd  # pandas için eklendi

app = Flask('app')
app.secret_key = config('secret')
my_client = pymongo.MongoClient(config('mongo_url'))
my_db = my_client[config('db_name')]


def get_sequence(seq_name):
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


@app.route('/')
@login_required
def index():
    return render_template('main.html')


@app.route('/submit', methods=['POST'])
@login_required  # Bu decorator'ın tanımı, yalnızca giriş yapan kullanıcılar için submit işlemini sınırlandırır.
def submit():
    # Form verilerini al
    patient_data = {
        "name": request.form.get("name"),
        "age": int(request.form.get("age")),
        "weight": float(request.form.get("weight")),
        "height": float(request.form.get("height")),
        "bmi": float(request.form.get("bmi")),
        "sigara": "sigara" in request.form,
        "alkol": "alkol" in request.form,
        "drug": "drug" in request.form,
        "familyHistory": request.form.get("familyHistory"),
        "bloodPressure": request.form.get("bloodPressure")
    }

    # Veritabanına patientData ekleyin
    my_db.patientData.insert_one(patient_data)  # 'db' yerine 'my_db' kullanıldı

    # Excel dosyasını işle
    if 'bloodValues' in request.files:
        file = request.files['bloodValues']
        if file and file.filename.endswith('.xlsx'):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            # Excel dosyasını oku ve veritabanına ekle
            df = pd.read_excel(filepath)
            blood_values = df.to_dict(orient='records')  # DataFrame'i dictionary listesine dönüştür
            my_db.bloodValue.insert_many(blood_values)  # 'db' yerine 'my_db' kullanıldı
            os.remove(filepath)  # Dosyayı sil

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)