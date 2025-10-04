from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, make_response, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv
import joblib
import numpy as np
from flask import flash
import pandas as pd
import shap
import uuid
from datetime import datetime, timedelta, timezone
from keras.models import load_model
from PIL import Image
import io
from ml_utils import generate_shap_plot_base64, generate_insights
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import re
from email.message import EmailMessage
import smtplib
import ssl
from threading import Thread
import requests
from zipfile import ZipFile
from functools import wraps
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def download_and_extract(url, target_dir, zip_name):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, zip_name)

    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name} ...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Extracting {zip_name} ...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
    else:
        print(f"{zip_name} already downloaded.")

def ensure_models():
    models = {
        "tabular_heart_disease_model_v1.pkl": {
            "url": "https://github.com/RafeyCooper/Healthify/releases/download/v1.0.0/tabular_heart_disease_model_v1.zip",
            "zip_name": "tabular_heart_disease_model_v1.zip"
        },
        "tabular_kidney_disease_model_v1.pkl": {
            "url": "https://github.com/RafeyCooper/Healthify/releases/download/v1.0.0/tabular_kidney_disease_model_v1.zip",
            "zip_name": "tabular_kidney_disease_model_v1.zip"
        },
        "image_retinopathy_model_v1.h5": {
            "url": "https://github.com/RafeyCooper/Healthify/releases/download/v1.0.0/image_retinopathy_model_v1.zip",
            "zip_name": "image_retinopathy_model_v1.zip"
        },
        "image_chest_xray_model_v1.h5": {
            "url": "https://github.com/RafeyCooper/Healthify/releases/download/v1.0.0/image_chest_xray_model_v1.zip",
            "zip_name": "image_chest_xray_model_v1.zip"
        },
        "xray_feature_bank.pkl": {
            "url": "https://github.com/umair25591/Healthify/releases/download/v1.0.0/xray_feature_bank.zip",
            "zip_name": "xray_feature_bank.zip"
        }
    }

    target_dir = "model"

    for model_file, info in models.items():
        model_path = os.path.join(target_dir, model_file)
        if not os.path.exists(model_path):
            print(f"{model_file} missing, downloading...")
            download_and_extract(info["url"], target_dir, info["zip_name"])
        else:
            print(f"{model_file} already exists ✅")

try:
    ensure_models()

    heart_model = joblib.load("model/tabular_heart_disease_model_v1.pkl")
    kidney_model = joblib.load("model/tabular_kidney_disease_model_v1.pkl")
    retinopathy_model = load_model("model/image_retinopathy_model_v1.h5")
    chest_model = load_model("model/image_chest_xray_model_v1.h5")

    with open('model/xray_feature_bank.pkl', 'rb') as f:
        xray_feature_bank = pickle.load(f)


    heart_explainer = shap.TreeExplainer(heart_model)
    kidney_explainer = shap.TreeExplainer(kidney_model)

except Exception as e:
    heart_model = None
    kidney_model = None
    retinopathy_model = None
    chest_model = None
    print(f"⚠️ Model loading failed: {e}")

load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["MONGO_URI"] = os.getenv("MONGOURI")

# Email configuration
app.config['EMAIL_SMTP_SERVER'] = os.getenv('EMAIL_SMTP_SERVER')
app.config['EMAIL_SMTP_PORT'] = int(os.getenv('EMAIL_SMTP_PORT', '587'))
app.config['EMAIL_USERNAME'] = os.getenv('EMAIL_USERNAME')
app.config['EMAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')
app.config['EMAIL_FROM'] = os.getenv('EMAIL_FROM') or app.config['EMAIL_USERNAME']

# Doctor portal config (v1 minimal)
app.config['DOCTOR_PORTAL_SECRET'] = os.getenv('DOCTOR_PORTAL_SECRET', 'change-this-secret')
app.config['DOCTOR_PORTAL_USER'] = os.getenv('DOCTOR_PORTAL_USER', 'doctor')
app.config['DOCTOR_PORTAL_PASS'] = os.getenv('DOCTOR_PORTAL_PASS', 'doctor123')

# Admin portal config (v1 minimal)
app.config['ADMIN_PORTAL_USER'] = os.getenv('ADMIN_PORTAL_USER', 'admin')
app.config['ADMIN_PORTAL_PASS'] = os.getenv('ADMIN_PORTAL_PASS', 'admin123')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


mongo = PyMongo(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'home'

# --- Minimal specialist/doctor mapping for V1 bookings ---
SPECIALTIES = {
    'heart': {
        'name': 'Cardiologist',
        'doctor_name': os.getenv('DOCTOR_HEART_NAME', 'Dr. Sudani Khan'),
        'doctor_email': os.getenv('DOCTOR_HEART_EMAIL', os.getenv('DEFAULT_DOCTOR_EMAIL', ''))
    },
    'kidney': {
        'name': 'Nephrologist',
        'doctor_name': os.getenv('DOCTOR_KIDNEY_NAME', 'Dr. Sudani Khan'),
        'doctor_email': os.getenv('DOCTOR_KIDNEY_EMAIL', os.getenv('DEFAULT_DOCTOR_EMAIL', ''))
    },
    'eye': {
        'name': 'Ophthalmologist',
        'doctor_name': os.getenv('DOCTOR_EYE_NAME', 'Dr. Sudani Khan'),
        'doctor_email': os.getenv('DOCTOR_EYE_EMAIL', os.getenv('DEFAULT_DOCTOR_EMAIL', ''))
    },
    'chest': {
        'name': 'Pulmonologist',
        'doctor_name': os.getenv('DOCTOR_CHEST_NAME', 'Dr. Sudani Khan'),
        'doctor_email': os.getenv('DOCTOR_CHEST_EMAIL', os.getenv('DEFAULT_DOCTOR_EMAIL', ''))
    },
}


# --- Doctor portal helpers ---
def doctor_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('doctor_logged_in'):
            return redirect(url_for('doctor_login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

class PDF(FPDF):
    THEME_PRIMARY = (59, 130, 246)
    THEME_TEXT = (17, 24, 39)
    THEME_MUTED = (75, 85, 99)
    THEME_LIGHT = (249, 250, 251)

    def header(self):
        # Top brand bar
        self.set_fill_color(*self.THEME_PRIMARY)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 12, 'Healthify AI - Medical Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C', fill=True)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*self.THEME_MUTED)
        self.cell(0, 10, f'Page {self.page_no()}', align='L')
        self.set_x(-55)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='R')

    def chapter_title(self, title):
        # Styled section header
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*self.THEME_TEXT)
        self.set_fill_color(243, 244, 246)  # Gray-100
        self.cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L', fill=True)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*self.THEME_TEXT)
        self.multi_cell(0, 5, body)
        self.ln()
        
    def key_value_pair(self, key, value):
        # Two-column key/value row with subtle background
        self.set_fill_color(*self.THEME_LIGHT)
        self.set_text_color(*self.THEME_MUTED)
        self.set_font('Helvetica', 'B', 10)
        self.cell(55, 8, key, border=0, align='L', fill=True)
        self.set_text_color(*self.THEME_TEXT)
        self.set_font('Helvetica', '', 10)
        self.cell(0, 8, str(value), border=0, align='L', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def build_prediction_pdf(prediction):
    pdf = PDF()
    pdf.add_page()

    # If patient details are present, generate the image-style report; else tabular-style
    if prediction.get('patient_details'):
        # Patient Details
        pdf.chapter_title('Patient Information')
        patient_details = prediction.get('patient_details', {})
        pdf.key_value_pair('Patient Name:', patient_details.get('patientName', 'N/A'))
        pdf.key_value_pair('Age:', patient_details.get('patientAge', 'N/A'))
        pdf.key_value_pair('Gender:', patient_details.get('patientGender', 'N/A'))
        pdf.key_value_pair('Contact:', patient_details.get('patientContact', 'N/A'))
        pdf.key_value_pair('Email:', patient_details.get('patientEmail', 'N/A'))
        pdf.ln(5)

        # Analysis Details
        pdf.chapter_title('AI Analysis Details')
        is_positive = 'positive' in prediction.get('prediction_result', '').lower() or 'pneumonia' in prediction.get('prediction_result', '').lower()
        friendly_response = get_user_friendly_response(prediction.get('disease_type', 'N/A'), is_positive, prediction.get('confidence_probability', 0))

        pdf.key_value_pair('Analysis Type:', str(prediction.get('disease_type', 'N/A')).capitalize())
        try:
            pdf.key_value_pair('Prediction Date:', (prediction['timestamp'] + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"))
        except Exception:
            pdf.key_value_pair('Prediction Date:', 'N/A')
        pdf.key_value_pair('Result:', friendly_response.get('title', 'N/A'))
        pdf.key_value_pair('Confidence:', f"{prediction.get('confidence_probability', 'N/A')}% ({friendly_response.get('confidence_level', 'N/A')})")
        pdf.ln(5)

        # Interpretation
        pdf.chapter_title('Interpretation & Recommendations')
        pdf.chapter_body(f"What this means: {friendly_response.get('explanation', 'N/A')}")
        pdf.chapter_body(f"Next Steps: {friendly_response.get('next_steps', 'N/A')}")
        pdf.ln(10)

        # Disclaimer
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 10, 'Disclaimer', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        pdf.set_font('Helvetica', 'I', 9)
        pdf.chapter_body(friendly_response.get('disclaimer', ''))
    else:
        # Tabular-style report
        pdf.chapter_title('Prediction Details')
        # current_user is not available in background threads, so omit username here
        pdf.key_value_pair('Analysis Type:', str(prediction.get('disease_type', 'N/A')).capitalize() + " Disease")
        try:
            pdf.key_value_pair('Prediction Date:', (prediction['timestamp'] + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"))
        except Exception:
            pdf.key_value_pair('Prediction Date:', 'N/A')
        pdf.key_value_pair('Result:', f"{prediction.get('prediction_result')} ({prediction.get('confidence_probability')}%)")
        pdf.ln(5)

        # User Inputs
        pdf.chapter_title('User-Provided Information')
        for key, value in (prediction.get('input_features') or {}).items():
            if str(key).lower() != 'diseasetype':
                formatted_key = re.sub(r'(?<!^)(?=[A-Z])', ' ', str(key))
                pdf.key_value_pair(f'{formatted_key.title()}:', value)
        pdf.ln(5)

        # Insights
        pdf.chapter_title('Actionable Health Insights')
        insights = prediction.get('insights', []) or []
        if insights:
            for insight in insights:
                clean_text = re.sub('<[^<]+?>', '', str(insight.get('text', '')))
                pdf.chapter_body(f"- {clean_text}")
        else:
            pdf.chapter_body("No specific insights were generated for this prediction.")
        pdf.ln(10)

        # Disclaimer
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 10, 'Disclaimer', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        pdf.set_font('Helvetica', 'I', 9)
        pdf.chapter_body("This is an AI-generated prediction and should not be considered medical advice. Always consult with a qualified healthcare professional for diagnosis and treatment.")

    return bytes(pdf.output(dest='S'))


def send_email_with_attachment(to_email, subject, body_text, attachment_bytes, attachment_filename):
    smtp_server = app.config.get('EMAIL_SMTP_SERVER')
    smtp_port = app.config.get('EMAIL_SMTP_PORT')
    smtp_user = app.config.get('EMAIL_USERNAME')
    smtp_pass = app.config.get('EMAIL_PASSWORD')
    from_email = app.config.get('EMAIL_FROM')
    smtp_debug = os.getenv('EMAIL_SMTP_DEBUG', '0') == '1'
    use_ssl = str(smtp_port) == '465' or os.getenv('EMAIL_USE_SSL', '0') == '1'

    if not smtp_server or not smtp_user or not smtp_pass or not from_email:
        print('Email not sent: missing SMTP configuration (EMAIL_SMTP_SERVER/EMAIL_USERNAME/EMAIL_PASSWORD/EMAIL_FROM).')
        return

    try:
        if 'gmail' in str(smtp_server).lower() and from_email and smtp_user and from_email != smtp_user:
            print('Warning: With Gmail, EMAIL_FROM should match EMAIL_USERNAME to avoid rejection.')

        message = EmailMessage()
        message['Subject'] = subject
        message['From'] = from_email
        message['To'] = to_email
        message.set_content(body_text)

        message.add_attachment(attachment_bytes, maintype='application', subtype='pdf', filename=attachment_filename)

        context = ssl.create_default_context()
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                if smtp_debug: server.set_debuglevel(1)
                server.login(smtp_user, smtp_pass)
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_debug: server.set_debuglevel(1)
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.send_message(message)
        print(f"Email sent to {to_email} with attachment {attachment_filename}")
    except smtplib.SMTPAuthenticationError as auth_error:
        try:
            detail = getattr(auth_error, 'smtp_error', b'').decode('utf-8', errors='ignore')
        except Exception:
            detail = str(auth_error)
        print(f"Failed to send email: SMTPAuthenticationError - {detail}")
    except Exception as email_error:
        print(f"Failed to send email to {to_email}: {email_error}")


def send_plain_email(to_email, subject, body_text):
    smtp_server = app.config.get('EMAIL_SMTP_SERVER')
    smtp_port = app.config.get('EMAIL_SMTP_PORT')
    smtp_user = app.config.get('EMAIL_USERNAME')
    smtp_pass = app.config.get('EMAIL_PASSWORD')
    from_email = app.config.get('EMAIL_FROM')
    smtp_debug = os.getenv('EMAIL_SMTP_DEBUG', '0') == '1'
    use_ssl = str(smtp_port) == '465' or os.getenv('EMAIL_USE_SSL', '0') == '1'

    if not smtp_server or not smtp_user or not smtp_pass or not from_email:
        print('Plain email not sent: missing SMTP configuration.')
        return

    try:
        message = EmailMessage()
        message['Subject'] = subject
        message['From'] = from_email
        message['To'] = to_email
        message.set_content(body_text)

        context = ssl.create_default_context()
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                if smtp_debug: server.set_debuglevel(1)
                server.login(smtp_user, smtp_pass)
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_debug: server.set_debuglevel(1)
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.send_message(message)
        print(f"Email sent to {to_email}")
    except Exception as email_error:
        print(f"Failed to send plain email to {to_email}: {email_error}")


def _send_prediction_pdf_task(prediction, recipient_email):
    try:
        if not recipient_email:
            print('No recipient email provided; skipping email send.')
            return
        pdf_bytes = build_prediction_pdf(prediction)
        disease_text = str(prediction.get('disease_type', 'Prediction')).capitalize()
        subject = f"Your Healthify AI Report - {disease_text}"
        body = (
            f"Hello,\n\nYour {disease_text} report is attached.\n"
            f"Result: {prediction.get('prediction_result')}\n"
            f"Regards,\nHealthify AI"
        )
        filename = f"report_{prediction.get('_id', uuid.uuid4().hex)}.pdf"
        send_email_with_attachment(recipient_email, subject, body, pdf_bytes, filename)
    except Exception as task_error:
        print(f"Error generating/sending prediction PDF email: {task_error}")


def enqueue_send_prediction_pdf(prediction, recipient_email):
    Thread(target=_send_prediction_pdf_task, args=(prediction, recipient_email), daemon=True).start()

def generate_meeting_link():
    room_name = f"healthify-{uuid.uuid4().hex}"
    return f"https://meet.jit.si/{room_name}"

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.username = user_data["username"]
        self.email = user_data["email"]
        self.password_hash = user_data["password"]

    @staticmethod
    def get(user_id):
        user_data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(user_data)
        return None
    
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

def preprocess_heart_data_for_df(data):
    try:
        age = int(data.get('AgeCategory', 0))
        if 18 <= age <= 24: age_category = 1;
        elif 25 <= age <= 29: age_category = 2
        elif 30 <= age <= 34: age_category = 3
        elif 35 <= age <= 39: age_category = 4
        elif 40 <= age <= 44: age_category = 5
        elif 45 <= age <= 49: age_category = 6
        elif 50 <= age <= 54: age_category = 7
        elif 55 <= age <= 59: age_category = 8
        elif 60 <= age <= 64: age_category = 9
        elif 65 <= age <= 69: age_category = 10
        elif 70 <= age <= 74: age_category = 11
        elif 75 <= age <= 79: age_category = 12
        else: age_category = 13

        gen_health_map = {'Excellent': 1, 'Very good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
        gen_health = gen_health_map.get(data.get('GenHealth'), 3)

        bmi = float(data.get('BMI', 0))
        sleep_time = int(data.get('SleepTime', 0))
        physical_health = int(data.get('PhysicalHealth', 0))
        mental_health = int(data.get('MentalHealth', 0))

        return [bmi, physical_health, mental_health, sleep_time, age_category, gen_health]
    except (ValueError, TypeError):
        return None

def preprocess_kidney_data_for_df(data):
    try:
        heart_disease_map = {'No': 0, 'Yes': 1}
        diff_walking_map = {'No': 0, 'Yes': 1}
        age_map = {
            '18-24': 9, '25-29': 12, '30-34': 11, '35-39': 10, '40-44': 4,
            '45-49': 8, '50-54': 7, '55-59': 0, '60-64': 6, '65-69': 2,
            '70-74': 5, '75-79': 3, '80 or older': 1
        }
        race_map = {
            'White': 0, 'Black': 1, 'Asian': 2, 
            'American Indian/Alaskan Native': 3, 'Other': 4, 'Hispanic': 5
        }
        diabetic_map = {
            'No': 1, 'Yes': 0, 'No, borderline diabetes': 2, 'Yes (during pregnancy)': 3
        }
        gen_health_map = {
            'Excellent': 4, 'Very good': 0, 'Good': 2, 'Fair': 1, 'Poor': 3
        }

        bmi = float(data.get('BMI', 0))
        physical_health = int(data.get('PhysicalHealth', 0))
        mental_health = int(data.get('MentalHealth', 0))
        sleep_time = int(data.get('SleepTime', 0))
        
        heart_disease = heart_disease_map.get(data.get('HeartDisease'), 0)
        diff_walking = diff_walking_map.get(data.get('DiffWalking'), 0)
        age_category = age_map.get(data.get('AgeCategory'), 0)
        race = race_map.get(data.get('Race'), 0)
        diabetic = diabetic_map.get(data.get('Diabetic'), 1)
        gen_health = gen_health_map.get(data.get('GenHealth'), 2)

        return [
            sleep_time,
            age_category,
            bmi,
            physical_health,
            mental_health,
            gen_health,
            diabetic,
            diff_walking,
            race,
            heart_disease
        ]
        
    except (ValueError, TypeError) as e:
        print(f"Error during kidney data preprocessing: {e}")
        return None

def preprocess_image(image_file, target_size=(150, 150)):
    try:
        # Reset stream position to the beginning before reading
        image_file.stream.seek(0)
        img = Image.open(image_file.stream).convert('RGB')
        img = img.resize(target_size)
        img_array = np.asarray(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def get_user_friendly_response(analysis_type, is_positive, confidence_percent):
                response = {
                    "disclaimer": "This is an AI-powered screening tool, not a medical diagnosis. Always consult a qualified healthcare professional."
                }
                
                if confidence_percent > 90:
                    response["confidence_level"] = "High"
                elif confidence_percent > 70:
                    response["confidence_level"] = "Medium"
                else:
                    response["confidence_level"] = "Low"

                if analysis_type == 'retinopathy':
                    if is_positive:
                        response["title"] = "Potential Signs of Retinopathy Detected"
                        response["explanation"] = "Our AI analysis identified patterns that may be associated with retinopathy."
                        response["next_steps"] = "We strongly recommend sharing this result with an ophthalmologist for a formal diagnosis."
                        response["status"] = "positive"
                    else:
                        response["title"] = "No Clear Signs of Retinopathy Detected"
                        response["explanation"] = "Our AI analysis did not find common indicators of retinopathy in the provided image."
                        response["next_steps"] = "Continue with your regular health check-ups as advised by your doctor."
                        response["status"] = "negative"
                
                elif analysis_type == 'xray':
                    if is_positive:
                        response["title"] = "Potential Signs of Pneumonia Detected"
                        response["explanation"] = "The AI analysis identified features in the chest X-ray that are commonly associated with pneumonia."
                        response["next_steps"] = "Please consult a radiologist or your primary care physician for a comprehensive evaluation and diagnosis."
                        response["status"] = "positive"
                    else:
                        response["title"] = "No Clear Signs of Pneumonia Detected"
                        response["explanation"] = "The AI analysis did not find common indicators of pneumonia in the provided chest X-ray."
                        response["next_steps"] = "Continue to follow your doctor's advice regarding your health."
                        response["status"] = "negative"
                
                return response

def extract_feature(img_path_or_stream, target_size=(224, 224)):
    if hasattr(img_path_or_stream, 'read'):
        img_path_or_stream.seek(0)
        img = Image.open(img_path_or_stream).convert('RGB')
    else:
        img = Image.open(img_path_or_stream).convert('RGB')

    img = img.resize(target_size)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = xray_feature_extractor.predict(x)
    return feat[0]


xray_feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def is_valid_xray_image(img_file, threshold=0.6):
    if not xray_feature_bank:
        return True

    query_feat = extract_feature(img_file)

    similarities = [
        cosine_similarity([query_feat], [ref_feat])[0][0]
        for ref_feat in xray_feature_bank.values()
    ]
    max_sim = max(similarities) if similarities else 0

    print(f"🔍 Max X-ray similarity: {max_sim:.3f}")

    return max_sim >= threshold


@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template("index.html")


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"message": "Missing username, email, or password"}), 400

    if mongo.db.users.find_one({"email": email}):
        return jsonify({"message": "Email address already in use."}), 409
    if mongo.db.users.find_one({"username": username}):
        return jsonify({"message": "Username already taken."}), 409
        
    hashed_password = generate_password_hash(password)
    mongo.db.users.insert_one({
        "username": username,
        "email": email,
        "password": hashed_password
    })
    
    return jsonify({"message": "Signup successful! Please log in."}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"message": "Missing username or password"}), 400

    user_data = mongo.db.users.find_one({"username": username})

    if user_data and check_password_hash(user_data['password'], password):
        user = User(user_data)
        login_user(user)
        return jsonify({"message": "Login successful!", "redirect": url_for('dashboard')}), 200
    
    return jsonify({"message": "Invalid username or password."}), 401


@app.route('/login-page')
def loginPage():
    return render_template("login.html")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


# --- Dashboard Section Routes ---
@app.route('/new_prediction')
@login_required
def new_prediction():
    return render_template('dashboard/new_prediction.html')

@app.route('/history')
@login_required
def history():
    user_predictions = list(mongo.db.predictions.find({
        'user_id': current_user.id
    }).sort('timestamp', -1))

    for pred in user_predictions:
        pred['timestamp'] = pred['timestamp'] + timedelta(hours=5)
    
    return render_template('dashboard/history.html', history=user_predictions)

@app.route('/history/<prediction_id>')
@login_required
def prediction_details(prediction_id):
    try:
        prediction = mongo.db.predictions.find_one({
            '_id': ObjectId(prediction_id),
            'user_id': current_user.id 
        })

        if not prediction:
            abort(404)

        prediction['timestamp'] = prediction['timestamp'] + timedelta(hours=5)
            
        return render_template('dashboard/prediction_details.html', prediction=prediction)
    except Exception as e:
        print(f"Error fetching prediction details: {e}")
        abort(404)

@app.route('/profile')
@login_required
def profile():
    return render_template('dashboard/profile.html')

@app.route('/tabular_prediction', methods=['GET', 'POST'])
@login_required
def tabular_prediction():
    if request.method == 'GET':
        return render_template('dashboard/tabular_prediction.html')

    if request.method == 'POST':
        data = request.form.to_dict()
        disease_type = data.get('diseaseType')

        model_configs = {
            'heart': {
                'model': heart_model,
                'explainer': heart_explainer,
                'preprocess_func': preprocess_heart_data_for_df,
                'feature_names': ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory', 'GenHealth']
            },
            'kidney': {
                'model': kidney_model,
                'explainer': kidney_explainer,
                'preprocess_func': preprocess_kidney_data_for_df,
                'feature_names': ['SleepTime', 'AgeCategory', 'BMI', 'PhysicalHealth', 'MentalHealth', 'GenHealth', 'Diabetic', 'DiffWalking', 'Race', 'HeartDisease']
            }
        }

        config = model_configs.get(disease_type)
        if not config or not config.get('model'):
            flash(f'{disease_type.capitalize()} disease model is not available.', 'warning')
            return redirect(url_for('tabular_prediction'))

        processed_data = config['preprocess_func'](data)
        if not processed_data:
            flash('Error processing your data. Please check your inputs.', 'danger')
            return redirect(url_for('tabular_prediction'))

        prediction_df = pd.DataFrame([processed_data], columns=config['feature_names'])
        
        probabilities = config['model'].predict_proba(prediction_df)
        high_risk_prob = probabilities[0][1]
        
        THRESHOLD = 0.35
        result_text = "High Risk" if high_risk_prob >= THRESHOLD else "Low Risk"

        # Determine the doctor URL based on disease type for high-risk cases
        doctor_url = None
        if result_text == "High Risk":
            if disease_type == 'heart':
                doctor_url = url_for('heart_doctor')
            elif disease_type == 'kidney':
                doctor_url = url_for('kidney_doctor')

        shap_image_base64, shap_values = generate_shap_plot_base64(config['explainer'], prediction_df)
        insights = generate_insights(shap_values, config['feature_names'], data)

        prediction_record = {
            'user_id': current_user.id,
            'disease_type': disease_type,
            'input_features': data,
            'prediction_result': result_text,
            'confidence_probability': int(high_risk_prob * 100),
            'timestamp': datetime.utcnow(),
            'insights': insights
        }
        inserted = mongo.db.predictions.insert_one(prediction_record)

        # Email report to the logged-in user in background
        try:
            prediction_doc = mongo.db.predictions.find_one({'_id': inserted.inserted_id})
            enqueue_send_prediction_pdf(prediction_doc, current_user.email)
        except Exception as _e:
            print(f"Failed to enqueue email for prediction {_e}")

        return render_template('dashboard/prediction_result.html',
                               disease=disease_type.capitalize(),
                               result=result_text,
                               probability=int(high_risk_prob * 100),
                               shap_image=shap_image_base64,
                               insights=insights,
                               prediction_id=str(inserted.inserted_id),
                               doctor_url=doctor_url)

    return redirect(url_for('tabular_prediction'))


@app.route('/image_prediction', methods=['GET', 'POST'])
@login_required
def image_prediction():
    if request.method == 'POST':
        image_file = request.files.get('image')
        analysis_type = request.form.get('type')
        patient_details = {
            'patientName': request.form.get('patientName'),
            'patientContact': request.form.get('patientContact'),
            'patientEmail': request.form.get('patientEmail'),
            'patientAge': request.form.get('patientAge'),
            'patientGender': request.form.get('patientGender')
        }

        if not image_file or image_file.filename == '':
            return jsonify({'error': 'No image file provided'}), 400
        if not analysis_type:
            return jsonify({'error': 'No analysis type specified'}), 400
        if not patient_details.get('patientName') or not patient_details.get('patientAge') or not patient_details.get('patientGender'):
            return jsonify({'error': 'Missing required patient details (Name, Age, Gender)'}), 400
        
        if analysis_type == 'xray':
            image_file.stream.seek(0)
            if not is_valid_xray_image(image_file):
                return jsonify({'error': 'The uploaded image does not appear to be a valid chest X-ray. Please upload a proper X-ray image.'}), 400
            image_file.stream.seek(0)

        filename = secure_filename(image_file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(filepath)

        if analysis_type == 'retinopathy':
            model = retinopathy_model
            target_size = (150, 150)
            class_labels = ['Negative', 'Positive']
        elif analysis_type == 'xray':
            model = chest_model
            target_size = (224, 224)
            class_labels = ['NORMAL', 'PNEUMONIA']
        else:
            return jsonify({'error': f'{analysis_type.capitalize()} model is not available.'}), 503

        processed_image = preprocess_image(image_file, target_size=target_size)
        if processed_image is None:
            return jsonify({'error': 'Failed to process the image.'}), 500

        try:
            prediction = model.predict(processed_image)
            confidence_score = float(prediction[0][0])
            THRESHOLD = 0.5

            is_positive = confidence_score >= THRESHOLD
            
            if is_positive:
                result_text = class_labels[1]
                confidence_percent = int(confidence_score * 100)
            else:
                result_text = class_labels[0]
                confidence_percent = int((1 - confidence_score) * 100)

            user_response = get_user_friendly_response(analysis_type, is_positive, confidence_percent)
            user_response['raw_confidence'] = confidence_percent
            user_response['patient_details'] = patient_details

            prediction_record = {
                'user_id': current_user.id,
                'disease_type': analysis_type,
                'patient_details': patient_details,
                'input_features': {'image_path': f'uploads/{unique_filename}'},
                'prediction_result': result_text,
                'confidence_probability': confidence_percent,
                'timestamp': datetime.utcnow()
            }
            inserted = mongo.db.predictions.insert_one(prediction_record)
            user_response['prediction_id'] = str(inserted.inserted_id) # Add prediction ID to response

            # Email report to the logged-in user in background
            try:
                prediction_doc = mongo.db.predictions.find_one({'_id': inserted.inserted_id})
                enqueue_send_prediction_pdf(prediction_doc, current_user.email)
            except Exception as _e:
                print(f"Failed to enqueue email for prediction {_e}")

            print(f'[USER: {current_user.username}] Prediction for {analysis_type}: {result_text} ({confidence_percent}%) -> Patient: {patient_details.get("patientName")}')

            return jsonify(user_response)

        except Exception as e:
            print(f"Error during model prediction: {e}")
            return jsonify({'error': 'An error occurred during analysis.'}), 500
        
    return render_template("dashboard/image_prediction.html")

@app.route('/download_report/<prediction_id>')
@login_required
def download_report(prediction_id):
    try:
        prediction = mongo.db.predictions.find_one({
            '_id': ObjectId(prediction_id),
            'user_id': current_user.id
        })
        if not prediction:
            abort(404)

        pdf_output = build_prediction_pdf(prediction)
        response = make_response(pdf_output)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=report_{prediction_id}.pdf'
        
        return response

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        abort(500)

@app.route('/download_tabular_report/<prediction_id>')
@login_required
def download_tabular_report(prediction_id):
    try:
        prediction = mongo.db.predictions.find_one({
            '_id': ObjectId(prediction_id),
            'user_id': current_user.id
        })
        if not prediction:
            abort(404)

        pdf_output = build_prediction_pdf(prediction)
        response = make_response(pdf_output)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=tabular_report_{prediction_id}.pdf'
        
        return response

    except Exception as e:
        print(f"Error generating tabular PDF report: {e}")
        abort(500)


@app.route('/dashboard')
@login_required
def dashboard():

    all_predictions = list(mongo.db.predictions.find({
        'user_id': current_user.id
    }).sort('timestamp', -1))

    for pred in all_predictions:
        pred['timestamp'] = pred['timestamp'] + timedelta(hours=5)

    recent_predictions = all_predictions[:3]

    positive_count = 0
    negative_count = 0
    for pred in all_predictions:
        result = pred.get('prediction_result', '').lower()
        if 'high' in result or 'positive' in result or 'pneumonia' in result:
            positive_count += 1
        else:
            negative_count += 1
            
    chart_data = {
        'positive': positive_count,
        'negative': negative_count
    }

    return render_template(
        'dashboard/dashboard.html', 
        recent_predictions=recent_predictions, 
        chart_data=chart_data
    )
    
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/heartds', methods=['GET', 'POST'])
def heartds():
    if request.method == 'POST':
        data = request.form.to_dict()
        return "OK", 200
    return render_template("heartds.html")

@app.route('/imagedata', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        data = request.form.to_dict()
        return "OK", 200
    return render_template("imagedata.html")

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    new_username = request.form.get('username')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    update_fields = {}
    
    if new_username and new_username != current_user.username:
        existing_user = mongo.db.users.find_one({"username": new_username})
        if existing_user:
            flash('Username already taken. Please choose another.', 'danger')
            return redirect(url_for('profile'))
        update_fields['username'] = new_username

    if new_password:
        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return redirect(url_for('profile'))
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return redirect(url_for('profile'))
            
        update_fields['password'] = generate_password_hash(new_password)

    if update_fields:
        mongo.db.users.update_one(
            {'_id': ObjectId(current_user.id)},
            {'$set': update_fields}
        )
        flash('Profile updated successfully!', 'success')
        if 'username' in update_fields:
            user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
            login_user(User(user_data))

    else:
        flash('No changes were made.', 'info')

    return redirect(url_for('profile'))

@app.route('/heart_doctor')
def heart_doctor():
    return render_template('dashboard/heart_doctor.html')

@app.route('/eye_doctor')
def eye_doctor():
    return render_template('dashboard/eye_doctor.html')

@app.route('/kidney_doctor')
def kidney_doctor():
    return render_template('dashboard/kidney_doctor.html')

@app.route('/chest_doctor')
def chest_doctor():
    return render_template('dashboard/chest_doctor.html')

@app.route('/doctorsuggestion')
def doctor_suggestion():
    return render_template('dashboard/doctorsuggestion.html')


@app.route('/book_appointment', methods=['GET', 'POST'])
@login_required
def book_appointment():
    if request.method == 'GET':
        specialty_key = request.args.get('specialty')
        prediction_id = request.args.get('prediction_id')
        specialties = get_specialties_from_db()
        specialty = specialties.get(specialty_key) if specialty_key else None

        # If prediction_id provided, prefill brief context
        prediction = None
        if prediction_id:
            try:
                prediction = mongo.db.predictions.find_one({
                    '_id': ObjectId(prediction_id),
                    'user_id': current_user.id
                })
            except Exception:
                prediction = None

        return render_template('dashboard/book_appointment.html',
                               specialties=specialties,
                               selected_specialty_key=specialty_key,
                               selected_specialty=specialty,
                               prediction=prediction)

    # POST - create appointment
    data = request.form
    specialty_key = data.get('specialty')
    doctor_id = data.get('doctor_id')
    appointment_date = data.get('date')  # YYYY-MM-DD
    appointment_time = data.get('time')  # HH:MM
    notes = data.get('notes')
    prediction_id = data.get('prediction_id')

    specialties = get_specialties_from_db()
    if not specialty_key or specialty_key not in specialties:
        flash('Please select a valid specialty.', 'danger')
        return redirect(url_for('book_appointment'))

    if not doctor_id:
        flash('Please select a doctor.', 'danger')
        return redirect(url_for('book_appointment', specialty=specialty_key, prediction_id=prediction_id))

    if not appointment_date or not appointment_time:
        flash('Please choose date and time.', 'danger')
        return redirect(url_for('book_appointment', specialty=specialty_key, prediction_id=prediction_id))

    try:
        start_dt = datetime.strptime(f"{appointment_date} {appointment_time}", '%Y-%m-%d %H:%M')
        end_dt = start_dt + timedelta(minutes=30)
    except Exception:
        flash('Invalid date/time format.', 'danger')
        return redirect(url_for('book_appointment', specialty=specialty_key, prediction_id=prediction_id))

    # Get doctor details
    doctor = mongo.db.doctors.find_one({'_id': ObjectId(doctor_id)})
    if not doctor:
        flash('Selected doctor not found.', 'danger')
        return redirect(url_for('book_appointment', specialty=specialty_key, prediction_id=prediction_id))

    # Check if the time slot is still available
    existing_appointment = mongo.db.appointments.find_one({
        'doctor_email': doctor['email'],
        'start_at': start_dt,
        'status': {'$in': ['confirmed', 'pending']}
    })
    
    if existing_appointment:
        flash('This time slot is no longer available. Please choose another time.', 'danger')
        return redirect(url_for('book_appointment', specialty=specialty_key, prediction_id=prediction_id))

    appointment_doc = {
        'user_id': current_user.id,
        'patient_name': current_user.username,     # add this
        'patient_email': current_user.email,
        'doctor_id': ObjectId(doctor_id),
        'specialty': specialty_key,
        'doctor_name': doctor['full_name'],
        'doctor_email': doctor['email'],
        'start_at': start_dt,
        'end_at': end_dt,
        'status': 'confirmed',
        'notes': notes or '',
        'created_at': datetime.utcnow(),
        'prediction_context_id': ObjectId(prediction_id) if prediction_id else None
    }
    inserted = mongo.db.appointments.insert_one(appointment_doc)

    # Notify doctor if email configured
    doctor_email = doctor['email']
    if doctor_email:
        try:
            summary = f"New appointment: {specialties[specialty_key]['name']}\n" \
                      f"Patient: {current_user.username} ({current_user.email})\n" \
                      f"Time: {start_dt.strftime('%Y-%m-%d %H:%M')}\n" \
                      f"Notes: {notes or 'N/A'}\n" \
                      f"Prediction ID: {prediction_id or 'N/A'}"
            send_plain_email(doctor_email, 'New Healthify Appointment', summary)
        except Exception as _e:
            print(f"Doctor email notification failed: {_e}")

    flash('Appointment booked successfully!', 'success')
    return redirect(url_for('appointments'))


@app.route('/appointments')
@login_required
def appointments():
    items = list(mongo.db.appointments.find({'user_id': current_user.id}).sort('start_at', -1))
    specialties = get_specialties_from_db()
    return render_template('dashboard/appointments.html', appointments=items, specialties=specialties, now=datetime.utcnow())


@app.route('/feedback/<appointment_id>', methods=['GET', 'POST'])
def feedback(appointment_id):
    appointment = mongo.db.appointments.find_one({"_id": ObjectId(appointment_id)})
    if not appointment:
        return render_template("feedback_message.html", message="Invalid appointment", type="info")

    existing = mongo.db.feedback.find_one({"appointment_id": ObjectId(appointment_id)})
    if existing:
        return render_template("feedback_message.html", message="Feedback already submitted for this appointment.", type="info")


    if request.method == 'POST':
        rating = int(request.form.get('rating', 0))
        comments = request.form.get('comments', '')

        mongo.db.feedback.insert_one({
            "appointment_id": ObjectId(appointment_id),
            "patient_email": appointment['patient_email'],
            "doctor_email": appointment['doctor_email'],
            "rating": rating,
            "comments": comments,
            "created_at": datetime.now(timezone.utc)
        })

        mongo.db.appointments.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": {"feedback_given": True}}
        )

        return render_template("feedback_message.html", message="Thank you for your feedback!", type="info")

    return render_template("feedback.html", appointment=appointment)


@app.route('/appointment/<appointment_id>/cancel', methods=['POST'])
@login_required
def cancel_appointment(appointment_id):
    try:
        appointment = mongo.db.appointments.find_one({
            '_id': ObjectId(appointment_id),
            'user_id': current_user.id # Security: Ensure user owns this appointment
        })

        if not appointment:
            flash('Appointment not found or you do not have permission to cancel it.', 'danger')
            return redirect(url_for('appointments'))

        # You can add logic here to prevent cancellation too close to the appointment time
        
        mongo.db.appointments.update_one(
            {'_id': ObjectId(appointment_id)},
            {'$set': {'status': 'cancelled'}}
        )

        # Optional: Notify the doctor of the cancellation via email
        # send_plain_email(appointment['doctor_email'], 'Appointment Cancelled', f"Patient {current_user.username} has cancelled their appointment on {appointment['start_at']}.")

        flash('Your appointment has been successfully cancelled.', 'success')
    except Exception as e:
        flash('An error occurred while cancelling the appointment.', 'danger')
        print(f"Error cancelling appointment: {e}")

    return redirect(url_for('appointments'))


# Add this route for the "View Details" button on past appointments
@app.route('/appointment/<appointment_id>/details')
@login_required
def appointment_details(appointment_id):
    try:
        appointment = mongo.db.appointments.find_one({
            '_id': ObjectId(appointment_id),
            'user_id': current_user.id
        })

        if not appointment:
            abort(404)
        
        # You'll need to create a new template: 'dashboard/appointment_details.html'
        # This template will display all appointment info, including doctor's notes, diagnosis, etc.
        return render_template('dashboard/appointment_details.html', appointment=appointment)

    except Exception as e:
        print(f"Error fetching appointment details: {e}")
        abort(500)


# --- Doctor Portal (MVP) ---
@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'GET':
        return render_template('Doctor/login.html')

    email = request.form.get('email')
    password = request.form.get('password')
    
    if not email or not password:
        flash('Please provide both email and password.', 'danger')
        return redirect(url_for('doctor_login'))

    # Find doctor by email
    doctor = mongo.db.doctors.find_one({'email': email.lower().strip()})
    
    if doctor and check_password_hash(doctor['password'], password):
        session['doctor_logged_in'] = True
        session['doctor_id'] = str(doctor['_id'])
        session['doctor_email'] = doctor['email']
        session['doctor_name'] = doctor['full_name']
        return redirect(url_for('doctor_dashboard'))
    
    flash('Invalid credentials.', 'danger')
    return redirect(url_for('doctor_login'))


@app.route('/doctor/logout')
def doctor_logout():
    session.pop('doctor_logged_in', None)
    session.pop('doctor_id', None)
    session.pop('doctor_email', None)
    session.pop('doctor_name', None)
    return redirect(url_for('doctor_login'))


@app.route('/doctor')
@doctor_login_required
def doctor_dashboard():
    # Get logged-in doctor's email
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return redirect(url_for('doctor_login'))
    
    # Get current time
    now = datetime.utcnow()
    
    # Get doctor's upcoming appointments
    items = list(mongo.db.appointments.find({
        'doctor_email': doctor_email,
        'start_at': {'$gte': now}
    }).sort('start_at', 1))
    
    specialties = get_specialties_from_db()
    return render_template('Doctor/dashboard.html', appointments=items, specialties=specialties)


@app.route('/doctor/schedule')
@doctor_login_required
def schedule_managment():
    return render_template('Doctor/schedule_managment.html')


@app.route('/doctor/patients')
@doctor_login_required
def doctor_patients():
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return redirect(url_for('doctor_login'))
    
    # Get search parameters
    search = request.args.get('search', '').strip()
    status_filter = request.args.get('status', 'all')
    specialty_filter = request.args.get('specialty', 'all')
    page = int(request.args.get('page', 1))
    per_page = 20
    
    # Build query for appointments
    query = {'doctor_email': doctor_email}
    
    if search:
        # Search in patient name, email, or phone
        query['$or'] = [
            {'patient_name': {'$regex': search, '$options': 'i'}},
            {'patient_email': {'$regex': search, '$options': 'i'}},
            {'patient_phone': {'$regex': search, '$options': 'i'}}
        ]
    
    if status_filter != 'all':
        query['status'] = status_filter
    
    if specialty_filter != 'all':
        query['specialty'] = specialty_filter
    
    # Get total count for pagination
    total_patients = mongo.db.appointments.count_documents(query)
    total_pages = (total_patients + per_page - 1) // per_page
    
    # Get appointments with pagination
    appointments = list(mongo.db.appointments.find(query)
                       .sort('start_at', -1)
                       .skip((page - 1) * per_page)
                       .limit(per_page))
    
    # Get unique patients (group by patient email)
    unique_patients = {}
    for apt in appointments:
        patient_key = apt.get('patient_email', apt.get('patient_name', 'Unknown'))
        if patient_key not in unique_patients:
            unique_patients[patient_key] = {
                'patient_name': apt.get('patient_name', 'Unknown'),
                'patient_email': apt.get('patient_email', ''),
                'patient_phone': apt.get('patient_phone', ''),
                'first_appointment': apt['start_at'],
                'last_appointment': apt['start_at'],
                'total_appointments': 1,
                'specialties': [apt.get('specialty', 'Unknown')],
                'statuses': [apt.get('status', 'Unknown')],
                'latest_appointment_id': str(apt['_id'])
            }
        else:
            unique_patients[patient_key]['total_appointments'] += 1
            unique_patients[patient_key]['last_appointment'] = apt['start_at']
            if apt.get('specialty') not in unique_patients[patient_key]['specialties']:
                unique_patients[patient_key]['specialties'].append(apt.get('specialty', 'Unknown'))
            if apt.get('status') not in unique_patients[patient_key]['statuses']:
                unique_patients[patient_key]['statuses'].append(apt.get('status', 'Unknown'))
    
    # Convert to list and sort by last appointment
    patients = list(unique_patients.values())
    patients.sort(key=lambda x: x['last_appointment'], reverse=True)
    
    # Get filter options
    all_specialties = list(mongo.db.appointments.distinct('specialty', {'doctor_email': doctor_email}))
    all_statuses = list(mongo.db.appointments.distinct('status', {'doctor_email': doctor_email}))

    # Count patients by status
    completed_count = sum(1 for p in patients if "completed" in p["statuses"])
    confirmed_count = sum(1 for p in patients if "confirmed" in p["statuses"])
    active_count = sum(1 for p in patients if "active" in p["statuses"])

    
    return render_template('Doctor/patients.html', 
                         patients=patients,
                         search=search,
                         status_filter=status_filter,
                         specialty_filter=specialty_filter,
                         page=page,
                         total_pages=total_pages,
                         total_patients=total_patients,
                         specialties=get_specialties_from_db(),
                         all_specialties=all_specialties,
                         all_statuses=all_statuses,
                         completed_count=completed_count,
                         confirmed_count=confirmed_count,
                         active_count=active_count
                         )


@app.route('/doctor/patient/<patient_email>')
@doctor_login_required
def doctor_patient_profile(patient_email):
    
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return redirect(url_for('doctor_login'))
    
    # Get all appointments for this patient with this doctor
    appointments = list(mongo.db.appointments.find({
        'doctor_email': doctor_email,
        'patient_email': patient_email
    }).sort('start_at', -1))
    
    if not appointments:
        flash('Patient not found.', 'danger')
        return redirect(url_for('doctor_patients'))
    
    # Get patient info from first appointment
    patient_info = {
        'name': appointments[0].get('patient_name', 'Unknown'),
        'email': patient_email,
        'phone': appointments[0].get('patient_phone', ''),
        'first_appointment': min(apt['start_at'] for apt in appointments),
        'last_appointment': max(apt['start_at'] for apt in appointments),
        'total_appointments': len(appointments),
        'specialties': list(set(apt.get('specialty', 'Unknown') for apt in appointments)),
        'statuses': list(set(apt.get('status', 'Unknown') for apt in appointments))
    }
    
    # Group appointments by status
    appointments_by_status = {}
    for apt in appointments:
        status = apt.get('status', 'Unknown')
        if status not in appointments_by_status:
            appointments_by_status[status] = []
        appointments_by_status[status].append(apt)
    
    # Get upcoming appointments
    now = datetime.utcnow()
    upcoming_appointments = [
        apt for apt in appointments
        if apt['start_at'] > now and apt.get('status') == 'confirmed'
    ]
    
    return render_template('Doctor/patient_profile.html',
                         patient=patient_info,
                         appointments=appointments,
                         appointments_by_status=appointments_by_status,
                         upcoming_appointments=upcoming_appointments,
                         specialties=get_specialties_from_db())

@app.route('/doctor/appointment/<appointment_id>')
@doctor_login_required
def doctor_appointment_detail(appointment_id):
    doctor_id = session.get('doctor_id')
    if not doctor_id:
        return redirect(url_for('doctor_login'))
    
    try:
        appointment = mongo.db.appointments.find_one({
            '_id': ObjectId(appointment_id),
            'doctor_id': ObjectId(doctor_id)
        })
        
        if not appointment:
            flash('Appointment not found.', 'danger')
            return redirect(url_for('doctor_dashboard'))
        
        patient_appointments = list(mongo.db.appointments.find({
            'doctor_id': ObjectId(doctor_id),
            'patient_email': appointment.get('patient_email'),
            '_id': {'$ne': ObjectId(appointment_id)}
        }).sort('start_at', -1).limit(5))
        
        patient_user_id = str(appointment['user_id'])
        conversation_id = get_conversation_id(doctor_id, patient_user_id)
        
        return render_template('Doctor/appointment_detail.html',
                             appointment=appointment,
                             patient_appointments=patient_appointments,
                             specialties=get_specialties_from_db(),
                             conversation_id=conversation_id)
    
    except Exception as e:
        print(f"Error loading appointment: {e}")
        flash('Error loading appointment details.', 'danger')
        return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/appointment/<appointment_id>/update', methods=['POST'])
@doctor_login_required
def doctor_appointment_update(appointment_id):
    
    doctor_email = session.get('doctor_email')
    
    if not doctor_email:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        notes = data.get('notes', '')
        diagnosis = data.get('diagnosis', '')
        prescription = data.get('prescription', '')
        
        
        # Update appointment
        result = mongo.db.appointments.update_one(
            {'_id': ObjectId(appointment_id), 'doctor_email': doctor_email},
            {'$set': {
                'doctor_notes': notes,
                'diagnosis': diagnosis,
                'prescription': prescription,
                'updated_at': datetime.utcnow()
            }}
        )
        
        if result.modified_count > 0:
            return jsonify({'success': True, 'message': 'Appointment updated successfully'})
        else:
            return jsonify({'success': False, 'message': 'Appointment not found or unauthorized'}), 404
    
    except Exception as e:
        print(f"Error updating appointment: {e}")
        return jsonify({'success': False, 'message': 'Failed to update appointment'}), 500

@app.route('/doctor/appointment/<appointment_id>/start', methods=['POST'])
@doctor_login_required
def start_appointment(appointment_id):
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401

    try:
        # Find appointment
        appointment = mongo.db.appointments.find_one({
            '_id': ObjectId(appointment_id),
            'doctor_email': doctor_email
        })

        if not appointment:
            return jsonify({'success': False, 'message': 'Appointment not found'}), 404

        # Generate meeting link (custom internal link)
        meeting_link = generate_meeting_link()

        # Update status -> active + save meeting link
        result = mongo.db.appointments.update_one(
            {'_id': ObjectId(appointment_id)},
            {'$set': {
                'status': 'active',
                'meeting_link': meeting_link,
                'started_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }}
        )

        if result.modified_count == 0:
            return jsonify({'success': False, 'message': 'No changes applied'}), 400

        # --- Send emails with meeting link ---
        subject = f"Your Appointment Has Started - {appointment['specialty'].title()}"

        body_patient = f"""
        Dear {appointment['patient_name']},

        Your appointment with {appointment['doctor_name']} has just started.

        📅 Date: {appointment['start_at'].strftime('%B %d, %Y')}
        ⏰ Time: {appointment['start_at'].strftime('%I:%M %p')} - {appointment['end_at'].strftime('%I:%M %p')}
        🩺 Specialty: {appointment['specialty'].title()}

        👉 Join Meeting: {meeting_link}

        Notes: {appointment.get('notes', 'N/A')}

        Regards,  
        Healthify
        """

        body_doctor = f"""
        Hello {appointment['doctor_name']},

        You have started the appointment with patient {appointment['patient_name']}.

        📅 Date: {appointment['start_at'].strftime('%B %d, %Y')}
        ⏰ Time: {appointment['start_at'].strftime('%I:%M %p')} - {appointment['end_at'].strftime('%I:%M %p')}
        👤 Patient Email: {appointment['patient_email']}

        👉 Meeting Link (share if needed): {meeting_link}

        Notes: {appointment.get('notes', 'N/A')}

        Regards,  
        Healthify
        """

        # Send to both
        send_plain_email(appointment['patient_email'], subject, body_patient)
        send_plain_email(appointment['doctor_email'], subject, body_doctor)

        return jsonify({'success': True, 'message': 'Appointment started, meeting link sent', 'meeting_link': meeting_link})

    except Exception as e:
        print(f"Error starting appointment: {e}")
        return jsonify({'success': False, 'message': 'Failed to start appointment'}), 500

@app.route('/doctor/appointment/<appointment_id>/complete', methods=['POST'])
@doctor_login_required
def complete_appointment(appointment_id):
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401

    try:
        # Find appointment
        appointment = mongo.db.appointments.find_one({
            '_id': ObjectId(appointment_id),
            'doctor_email': doctor_email
        })

        if not appointment:
            return jsonify({'success': False, 'message': 'Appointment not found'}), 404

        # Update status -> completed
        result = mongo.db.appointments.update_one(
            {'_id': ObjectId(appointment_id)},
            {'$set': {
                'status': 'completed',
                'completed_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }}
        )

        if result.modified_count == 0:
            return jsonify({'success': False, 'message': 'No changes applied'}), 400

        # --- Send emails ---
        subject = f"Appointment Completed - {appointment['specialty'].title()}"

        feedback_link = f"{request.host_url}feedback/{appointment_id}"

        body_patient = f"""
        Dear {appointment['patient_name']},

        Your appointment with {appointment['doctor_name']} has been marked as completed.

        📅 Date: {appointment['start_at'].strftime('%B %d, %Y')}
        ⏰ Time: {appointment['start_at'].strftime('%I:%M %p')} - {appointment['end_at'].strftime('%I:%M %p')}
        🩺 Specialty: {appointment['specialty'].title()}

        👉 Please take a moment to provide feedback: {feedback_link}

        Your feedback helps us improve our services. 💙

        Regards,
        Healthify
        """

        body_doctor = f"""
        Hello {appointment['doctor_name']},

        You have successfully completed the appointment with patient {appointment['patient_name']}.

        📅 Date: {appointment['start_at'].strftime('%B %d, %Y')}
        ⏰ Time: {appointment['start_at'].strftime('%I:%M %p')} - {appointment['end_at'].strftime('%I:%M %p')}
        👤 Patient Email: {appointment['patient_email']}

        Regards,
        Healthify
        """

        # Send to both
        send_plain_email(appointment['patient_email'], subject, body_patient)
        send_plain_email(appointment['doctor_email'], subject, body_doctor)

        return jsonify({'success': True, 'message': 'Appointment completed, feedback link sent', 'feedback_link': feedback_link})

    except Exception as e:
        print(f"Error completing appointment: {e}")
        return jsonify({'success': False, 'message': 'Failed to complete appointment'}), 500


@app.route('/doctor/patient/<patient_email>/history', methods=['GET', 'POST'])
@doctor_login_required
def doctor_patient_history(patient_email):
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return redirect(url_for('doctor_login'))
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            history_entry = {
                'patient_email': patient_email,
                'doctor_email': doctor_email,
                'date': datetime.utcnow(),
                'condition': data.get('condition', ''),
                'symptoms': data.get('symptoms', ''),
                'diagnosis': data.get('diagnosis', ''),
                'treatment': data.get('treatment', ''),
                'medications': data.get('medications', []),
                'notes': data.get('notes', ''),
                'follow_up_date': data.get('follow_up_date', ''),
                'created_at': datetime.utcnow()
            }
            
            # Upsert patient history
            mongo.db.patient_history.update_one(
                {'patient_email': patient_email, 'doctor_email': doctor_email},
                {'$push': {'entries': history_entry}},
                upsert=True
            )
            
            return jsonify({'success': True, 'message': 'Medical history updated'})
        
        except Exception as e:
            print(f"Error updating patient history: {e}")
            return jsonify({'success': False, 'message': 'Failed to update history'}), 500
    
    # GET request - return patient history
    try:
        patient_history = mongo.db.patient_history.find_one({
            'patient_email': patient_email,
            'doctor_email': doctor_email
        })
        
        if patient_history:
            # Sort entries by date (newest first)
            patient_history['entries'].sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            'success': True,
            'history': patient_history.get('entries', []) if patient_history else []
        })
    
    except Exception as e:
        print(f"Error loading patient history: {e}")
        return jsonify({'success': False, 'message': 'Failed to load history'}), 500


@app.route('/doctor/update-status', methods=['POST'])
@doctor_login_required
def doctor_update_status_ajax():
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    data = request.get_json()
    status = data.get('status')
    
    if status not in ['active', 'inactive']:
        return jsonify({'success': False, 'message': 'Invalid status'}), 400
    
    try:
        # Update doctor status in database
        mongo.db.doctors.update_one(
            {'email': doctor_email},
            {'$set': {'status': status, 'updated_at': datetime.utcnow()}}
        )
        
        return jsonify({'success': True, 'message': f'Status updated to {status}'})
    except Exception as e:
        print(f"Failed to update doctor status: {e}")
        return jsonify({'success': False, 'message': 'Failed to update status'}), 500


@app.route('/doctor/appointment/<appointment_id>/status', methods=['POST'])
@doctor_login_required
def doctor_update_status(appointment_id):
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return redirect(url_for('doctor_login'))
    
    new_status = request.form.get('status')
    if new_status not in ['confirmed', 'active', 'completed', 'cancelled', 'no_show']:
        flash('Invalid status.', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    try:
        # Update appointment status (only for doctor's own appointments)
        result = mongo.db.appointments.update_one(
            {'_id': ObjectId(appointment_id), 'doctor_email': doctor_email},
            {'$set': {'status': new_status, 'updated_at': datetime.utcnow()}}
        )
        
        if result.modified_count > 0:
            flash('Appointment updated.', 'success')
        else:
            flash('Appointment not found or unauthorized.', 'danger')
    except Exception as _e:
        print(f"Failed to update appointment: {_e}")
        flash('Update failed.', 'danger')
    
    return redirect(url_for('doctor_dashboard'))

# --- Admin Portal (MVP) ---
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'GET':
        return render_template('Admin/login.html')

    username = request.form.get('username')
    password = request.form.get('password')
    if username == app.config['ADMIN_PORTAL_USER'] and password == app.config['ADMIN_PORTAL_PASS']:
        session['admin_logged_in'] = True
        return redirect(url_for('admin_dashboard'))
    flash('Invalid credentials.', 'danger')
    return redirect(url_for('admin_login'))


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/feedback')
@admin_login_required
def admin_feedback():
    feedback_items = list(mongo.db.feedback.find({}).sort("created_at", -1))

    # Attach appointment info (start, end, started_at, completed_at)
    for fb in feedback_items:
        appointment = mongo.db.appointments.find_one({"_id": fb["appointment_id"]})
        fb["appointment"] = appointment

    return render_template('Admin/feedback.html', feedback=feedback_items)

@app.route('/admin/appointments')
@admin_login_required
def admin_appointments():
    search = request.args.get('search', '').strip()
    status = request.args.get('status', 'all')

    query = {}
    if search:
        try:
            # If search looks like an ObjectId, search by appointment ID
            query['_id'] = ObjectId(search)
        except:
            # Otherwise search patient/doctor
            query['$or'] = [
                {'patient_name': {'$regex': search, '$options': 'i'}},
                {'patient_email': {'$regex': search, '$options': 'i'}},
                {'doctor_name': {'$regex': search, '$options': 'i'}},
                {'doctor_email': {'$regex': search, '$options': 'i'}}
            ]

    if status != 'all':
        query['status'] = status

    appointments = list(mongo.db.appointments.find(query).sort('start_at', -1))

    return render_template(
        'Admin/appointments.html',
        appointments=appointments,
        search=search,
        status=status
    )

@app.route('/admin')
@admin_login_required
def admin_dashboard():
    # --- Stats ---
    total_appointments = mongo.db.appointments.count_documents({})
    total_doctors = mongo.db.doctors.count_documents({'status': 'active'})
    total_predictions = mongo.db.predictions.count_documents({})
    total_feedback = mongo.db.feedback.count_documents({})

    # --- Recent Appointments ---
    recent_appointments = list(mongo.db.appointments.find({}).sort('start_at', -1).limit(5))

    # --- Feedback ---
    recent_feedback = list(mongo.db.feedback.find({}).sort('created_at', -1).limit(5))

    # --- Appointments by Status ---
    status_counts = {
        'confirmed': mongo.db.appointments.count_documents({'status': 'confirmed'}),
        'active': mongo.db.appointments.count_documents({'status': 'active'}),
        'completed': mongo.db.appointments.count_documents({'status': 'completed'}),
        'cancelled': mongo.db.appointments.count_documents({'status': {'$in': ['cancelled', 'no_show']}})
    }

    # --- Popular Specialty ---
    pipeline = [
        {"$group": {"_id": "$specialty", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 1}
    ]
    top_specialty = list(mongo.db.appointments.aggregate(pipeline))
    most_popular_specialty = top_specialty[0] if top_specialty else None

    # --- Average Feedback Rating ---
    avg_rating_pipeline = [
        {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}
    ]
    avg_rating_result = list(mongo.db.feedback.aggregate(avg_rating_pipeline))
    avg_rating = round(avg_rating_result[0]['avg_rating'], 2) if avg_rating_result else None

    # --- Average Appointment Duration ---
    durations = []
    for appt in mongo.db.appointments.find({"started_at": {"$exists": True}, "completed_at": {"$exists": True}}):
        duration = (appt["completed_at"] - appt["started_at"]).total_seconds() / 60
        durations.append(duration)
    avg_duration = round(sum(durations) / len(durations), 1) if durations else None

    return render_template(
        'Admin/dashboard.html',
        total_appointments=total_appointments,
        total_doctors=total_doctors,
        total_predictions=total_predictions,
        total_feedback=total_feedback,
        recent_appointments=recent_appointments,
        recent_feedback=recent_feedback,
        status_counts=status_counts,
        most_popular_specialty=most_popular_specialty,
        avg_rating=avg_rating,
        avg_duration=avg_duration
    )


# --- Admin Doctor Management ---
@app.route('/admin/doctors')
@admin_login_required
def admin_doctors():
    doctors = list(mongo.db.doctors.find({}).sort('created_at', -1))
    return render_template('Admin/doctors.html', doctors=doctors, specialties=get_specialties_from_db())


@app.route('/admin/doctors/add', methods=['GET', 'POST'])
@admin_login_required
def admin_add_doctor():
    if request.method == 'GET':
        return render_template('Admin/add_doctor.html', specialties=get_specialties_from_db())
    
    # POST - add doctor
    data = request.form
    
    # Validate required fields
    required_fields = ['full_name', 'email', 'phone', 'specialty', 'license_number', 'experience_years', 'qualification', 'password', 'confirm_password']
    for field in required_fields:
        if not data.get(field):
            flash(f'{field.replace("_", " ").title()} is required.', 'danger')
            return redirect(url_for('admin_add_doctor'))

    # Validate password
    password = data.get('password')
    confirm_password = data.get('confirm_password')
    
    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('admin_add_doctor'))
    
    if len(password) < 6:
        flash('Password must be at least 6 characters long.', 'danger')
        return redirect(url_for('admin_add_doctor'))

    # Validate email format
    if not re.match(r"[^@]+@[^@]+\.[^@]+", data.get('email')):
        flash('Invalid email format.', 'danger')
        return redirect(url_for('admin_add_doctor'))

    # Validate phone format
    phone = data.get('phone').replace(' ', '').replace('-', '').replace('+', '')
    if not phone.isdigit() or len(phone) < 10:
        flash('Invalid phone number.', 'danger')
        return redirect(url_for('admin_add_doctor'))

    # Validate specialty
    specialties = get_specialties_from_db()
    if data.get('specialty') not in specialties:
        flash('Invalid specialty selected.', 'danger')
        return redirect(url_for('admin_add_doctor'))

    # Check if email already exists
    if mongo.db.doctors.find_one({'email': data.get('email')}):
        flash('Email already registered.', 'danger')
        return redirect(url_for('admin_add_doctor'))

    # Create doctor document
    doctor_doc = {
        'full_name': data.get('full_name').strip(),
        'email': data.get('email').lower().strip(),
        'phone': phone,
        'password': generate_password_hash(password),
        'specialty': data.get('specialty'),
        'license_number': data.get('license_number').strip(),
        'experience_years': int(data.get('experience_years')),
        'qualification': data.get('qualification').strip(),
        'bio': data.get('bio', '').strip(),
        'hospital': data.get('hospital', '').strip(),
        'consultation_fee': float(data.get('consultation_fee', 0)),
        'availability': data.get('availability', 'weekdays').strip(),
        'status': 'active',
        'created_at': datetime.utcnow(),
        'created_by': 'admin'
    }

    try:
        inserted = mongo.db.doctors.insert_one(doctor_doc)
        flash('Doctor added successfully!', 'success')
        return redirect(url_for('admin_doctors'))
    except Exception as e:
        print(f"Doctor addition failed: {e}")
        flash('Failed to add doctor. Please try again.', 'danger')
        return redirect(url_for('admin_add_doctor'))


@app.route('/admin/doctors/<doctor_id>')
@admin_login_required
def admin_doctor_details(doctor_id):
    try:
        doctor = mongo.db.doctors.find_one({'_id': ObjectId(doctor_id)})
        if not doctor:
            flash('Doctor not found.', 'danger')
            return redirect(url_for('admin_doctors'))
        
        # Get doctor's appointments
        appointments = list(mongo.db.appointments.find({'doctor_email': doctor['email']}).sort('start_at', -1).limit(10))
        
        return render_template('Admin/doctor_details.html', doctor=doctor, appointments=appointments, specialties=get_specialties_from_db())
    except Exception as e:
        flash('Error loading doctor details.', 'danger')
        return redirect(url_for('admin_doctors'))


@app.route('/admin/doctors/<doctor_id>/edit', methods=['GET', 'POST'])
@admin_login_required
def admin_edit_doctor(doctor_id):
    try:
        doctor = mongo.db.doctors.find_one({'_id': ObjectId(doctor_id)})
        if not doctor:
            flash('Doctor not found.', 'danger')
            return redirect(url_for('admin_doctors'))
        
        if request.method == 'GET':
            return render_template('Admin/edit_doctor.html', doctor=doctor, specialties=get_specialties_from_db())
        
        # POST - update doctor
        data = request.form
        
        # Validate required fields
        required_fields = ['full_name', 'email', 'phone', 'specialty', 'license_number', 'experience_years', 'qualification']
        for field in required_fields:
            if not data.get(field):
                flash(f'{field.replace("_", " ").title()} is required.', 'danger')
                return redirect(url_for('admin_edit_doctor', doctor_id=doctor_id))

        # Check if email already exists (excluding current doctor)
        existing_doctor = mongo.db.doctors.find_one({'email': data.get('email'), '_id': {'$ne': ObjectId(doctor_id)}})
        if existing_doctor:
            flash('Email already registered by another doctor.', 'danger')
            return redirect(url_for('admin_edit_doctor', doctor_id=doctor_id))

        # Update doctor document
        update_data = {
            'full_name': data.get('full_name').strip(),
            'email': data.get('email').lower().strip(),
            'phone': data.get('phone').replace(' ', '').replace('-', '').replace('+', ''),
            'specialty': data.get('specialty'),
            'license_number': data.get('license_number').strip(),
            'experience_years': int(data.get('experience_years')),
            'qualification': data.get('qualification').strip(),
            'bio': data.get('bio', '').strip(),
            'hospital': data.get('hospital', '').strip(),
            'consultation_fee': float(data.get('consultation_fee', 0)),
            'availability': data.get('availability', 'weekdays').strip(),
            'updated_at': datetime.utcnow()
        }

        mongo.db.doctors.update_one(
            {'_id': ObjectId(doctor_id)},
            {'$set': update_data}
        )
        
        flash('Doctor updated successfully!', 'success')
        return redirect(url_for('admin_doctor_details', doctor_id=doctor_id))
        
    except Exception as e:
        print(f"Doctor update failed: {e}")
        flash('Failed to update doctor.', 'danger')
        return redirect(url_for('admin_doctors'))


@app.route('/admin/doctors/<doctor_id>/delete', methods=['POST'])
@admin_login_required
def admin_delete_doctor(doctor_id):
    try:
        doctor = mongo.db.doctors.find_one({'_id': ObjectId(doctor_id)})
        if not doctor:
            flash('Doctor not found.', 'danger')
            return redirect(url_for('admin_doctors'))
        
        # Check if doctor has appointments
        appointment_count = mongo.db.appointments.count_documents({'doctor_email': doctor['email']})
        if appointment_count > 0:
            flash(f'Cannot delete doctor with {appointment_count} existing appointments.', 'danger')
            return redirect(url_for('admin_doctors'))
        
        mongo.db.doctors.delete_one({'_id': ObjectId(doctor_id)})
        flash('Doctor deleted successfully!', 'success')
        
    except Exception as e:
        print(f"Doctor deletion failed: {e}")
        flash('Failed to delete doctor.', 'danger')
    
    return redirect(url_for('admin_doctors'))

@app.route('/api/doctor/schedule/current', methods=['GET'])
@doctor_login_required
def get_current_schedule():
    try:
        doctor_email = session.get('doctor_email')
        if not doctor_email:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Get doctor's weekly schedule
        schedule = mongo.db.doctor_weekly_schedules.find_one({
            'doctor_email': doctor_email
        })
        
        return jsonify({
            'success': True,
            'schedule': schedule.get('schedule', {}) if schedule else {}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/doctor/schedule/weekly', methods=['POST'])
@doctor_login_required
def save_weekly_schedule():
   
    try:
        doctor_email = session.get('doctor_email')
        if not doctor_email:
            return jsonify({'error': 'Not authenticated'}), 401
        
        data = request.get_json()
        schedule_data = data
        
        # Update or create weekly schedule
        mongo.db.doctor_weekly_schedules.update_one(
            {'doctor_email': doctor_email},
            {
                '$set': {
                    'schedule': schedule_data,
                    'updated_at': datetime.utcnow()
                }
            },
            upsert=True
        )
        
        return jsonify({'success': True, 'message': 'Weekly schedule saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/doctor/schedule/daily', methods=['GET', 'POST'])
@doctor_login_required
def manage_daily_schedule():
    doctor_email = session.get('doctor_email')
    if not doctor_email:
        return jsonify({'error': 'Not authenticated'}), 401

    if request.method == 'GET':
        
        date = request.args.get('date')

        if not date:
            return jsonify({'error': 'Date required'}), 400
        
        target_date = datetime.strptime(date, '%Y-%m-%d')
        
        schedule = mongo.db.doctor_daily_schedules.find_one({
            'doctor_email': doctor_email,
            'date': target_date
        })
        
        return jsonify({
            'success': True,
            'schedule': schedule if schedule else {}
        })
    
    else:  # POST
        try:
            data = request.get_json()
            date = datetime.strptime(data['date'], '%Y-%m-%d')
            date = datetime.combine(date.date(), datetime.min.time())
            time_slots = data.get('timeSlots', [])
            is_available = data.get('isAvailable', True) # Get the availability flag

            # Prepare the update document
            update_doc = {
                'isAvailable': is_available,
                'time_slots': time_slots, # Save the list of available slots
                'updated_at': datetime.utcnow()
            }

            # Update or create the daily override
            mongo.db.doctor_daily_schedules.update_one(
                {
                    'doctor_email': doctor_email,
                    'date': date
                },
                {'$set': update_doc},
                upsert=True
            )
            
            return jsonify({'success': True, 'message': 'Daily schedule saved'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/doctor/schedule/bulk', methods=['POST'])
@doctor_login_required
def save_bulk_schedule():
    try:
        doctor_email = session.get('doctor_email')
        if not doctor_email:
            return jsonify({'error': 'Not authenticated'}), 401
        
        data = request.get_json()
        start_date = datetime.strptime(data['startDate'], '%Y-%m-%d')
        end_date = datetime.strptime(data['endDate'], '%Y-%m-%d')
        action = data.get('action') # 'unavailable' or 'reset'

        if not action:
            return jsonify({'error': 'Bulk action is required'}), 400

        # Loop through each day in the provided date range
        current_date = start_date
        while current_date <= end_date:
            if action == 'unavailable':
                # Set the day as unavailable with no time slots
                mongo.db.doctor_daily_schedules.update_one(
                    {'doctor_email': doctor_email, 'date': current_date},
                    {
                        '$set': {
                            'isAvailable': False,
                            'time_slots': [],
                            'updated_at': datetime.utcnow()
                        }
                    },
                    upsert=True
                )
            elif action == 'reset':
                # Delete the override document to revert to the weekly default
                mongo.db.doctor_daily_schedules.delete_one(
                    {'doctor_email': doctor_email, 'date': current_date}
                )
            
            current_date += timedelta(days=1)
        
        return jsonify({'success': True, 'message': 'Bulk schedule applied'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/doctors/available', methods=['GET'])
def get_available_doctors():
    try:
        # Get all active doctors
        doctors = list(mongo.db.doctors.find({
            'status': 'active'
        }, {
            '_id': 1,
            'full_name': 1,
            'specialty': 1,
            'status': 1,
            'qualification': 1,
            'experience_years': 1
        }))
        
        # Convert ObjectId to string for JSON serialization
        for doctor in doctors:
            doctor['_id'] = str(doctor['_id'])
        
        return jsonify({
            'success': True,
            'doctors': doctors
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/doctor/availability/<doctor_id>', methods=['GET'])
def get_doctor_availability(doctor_id):
    try:
        date = request.args.get('date')
        if not date:
            return jsonify({'error': 'Date required'}), 400
        
        target_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Get doctor's schedule for this date
        schedule = mongo.db.doctor_daily_schedules.find_one({
            'doctor_id': ObjectId(doctor_id),
            'date': target_date
        })
        
        # Get existing appointments for this date
        appointments = list(mongo.db.appointments.find({
            'doctor_id': ObjectId(doctor_id),
            'start_at': {
                '$gte': target_date,
                '$lt': target_date + timedelta(days=1)
            }
        }))
        
        # Generate available time slots
        available_slots = generate_available_slots(schedule, appointments)
        
        return jsonify({
            'success': True,
            'date': date,
            'available_slots': available_slots
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/book-appointment-quick', methods=['POST'])
@login_required
def book_appointment_quick():
    try:
        data = request.get_json()
        doctor_id = data.get('doctor_id')
        date = data.get('date')
        time_slot = data.get('time_slot')
        prediction_id = data.get('prediction_id')
        notes = data.get('notes', '')
        
        # Validate inputs
        if not all([doctor_id, date, time_slot]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if slot is still available
        target_date = datetime.strptime(date, '%Y-%m-%d')
        start_time = datetime.strptime(f"{date} {time_slot}", '%Y-%m-%d %H:%M')
        end_time = start_time + timedelta(minutes=30)
        
        # Check for conflicts
        existing_appointment = mongo.db.appointments.find_one({
            'doctor_id': ObjectId(doctor_id),
            'start_at': start_time,
            'status': {'$nin': ['cancelled', 'no_show']}
        })
        
        if existing_appointment:
            return jsonify({'error': 'Time slot no longer available'}), 409
        
        # Get doctor details
        doctor = mongo.db.doctors.find_one({'_id': ObjectId(doctor_id)})
        if not doctor:
            return jsonify({'error': 'Doctor not found'}), 404
        
        # Create appointment
        appointment_doc = {
            'user_id': current_user.id,
            'doctor_id': ObjectId(doctor_id),
            'doctor_email': doctor['email'],
            'doctor_name': doctor['full_name'],
            'specialty': doctor['specialty'],
            'start_at': start_time,
            'end_at': end_time,
            'status': 'confirmed',
            'notes': notes,
            'created_at': datetime.utcnow(),
            'prediction_context_id': ObjectId(prediction_id) if prediction_id else None
        }
        
        inserted = mongo.db.appointments.insert_one(appointment_doc)
        
        # Send notifications
        send_appointment_notifications(appointment_doc, doctor)
        
        return jsonify({
            'success': True,
            'appointment_id': str(inserted.inserted_id),
            'message': 'Appointment booked successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/doctor/appointments')
@doctor_login_required
def doctor_appointments():

    doctor_email = session.get("doctor_email")
    if not doctor_email:
        return redirect(url_for("doctor_login"))

    # Fetch different appointment categories
    now = datetime.utcnow()

    upcoming = list(mongo.db.appointments.find({
        "doctor_email": doctor_email,
        "start_at": {"$gte": now},
        "status": {"$in": ["confirmed", "active"]}
    }).sort("start_at", 1))

    completed = list(mongo.db.appointments.find({
        "doctor_email": doctor_email,
        "status": "completed"
    }).sort("start_at", -1))

    cancelled = list(mongo.db.appointments.find({
        "doctor_email": doctor_email,
        "status": {"$in": ["cancelled", "no_show"]}
    }).sort("start_at", -1))

    all_appts = list(mongo.db.appointments.find({
        "doctor_email": doctor_email
    }).sort("start_at", -1))

    specialties = get_specialties_from_db()

    return render_template(
        "Doctor/appointments_overview.html",
        upcoming=upcoming,
        completed=completed,
        cancelled=cancelled,
        all_appts=all_appts,
        specialties=specialties,
        now=now
    )

def generate_available_slots(schedule, appointments):
    # Default time slots (9 AM to 6 PM, 30-minute intervals)
    default_slots = [
        '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
        '12:00', '12:30', '14:00', '14:30', '15:00', '15:30',
        '16:00', '16:30', '17:00', '17:30'
    ]
    
    # Get doctor's available slots for this date
    available_slots = schedule.get('time_slots', default_slots) if schedule else default_slots
    
    # Remove booked slots
    booked_times = []
    for appointment in appointments:
        start_time = appointment['start_at'].strftime('%H:%M')
        end_time = appointment['end_at'].strftime('%H:%M')
        booked_times.extend([start_time, end_time])
    
    # Filter out booked slots
    final_slots = []
    for slot in available_slots:
        if slot not in booked_times:
            final_slots.append(slot)
    
    return final_slots

def send_appointment_notifications(appointment, doctor):
    try:
        # Email notification to doctor
        if doctor.get('email'):
            subject = 'New Healthify Appointment'
            body = f"""
            New appointment scheduled:
            
            Patient: {current_user.username} ({current_user.email})
            Date: {appointment['start_at'].strftime('%Y-%m-%d %I:%M %p')}
            Specialty: {appointment['specialty']}
            Notes: {appointment['notes'] or 'N/A'}
            """
            send_plain_email(doctor['email'], subject, body)
    except Exception as e:
        print(f"Failed to send appointment notifications: {e}")

def generate_time_slots(start_time, end_time, duration_minutes):
    slots = []
    start_dt = datetime.strptime(start_time, '%H:%M')
    end_dt = datetime.strptime(end_time, '%H:%M')
    
    current = start_dt
    while current < end_dt:
        slots.append(current.strftime('%H:%M'))
        current += timedelta(minutes=duration_minutes)
    
    return slots


# --- Specialty Management Routes ---

def get_specialties_from_db():
    try:
        specialties = {}
        for spec in mongo.db.specialties.find():
            specialties[spec['key']] = {
                'name': spec['name'],
                'description': spec.get('description', ''),
                'icon': spec.get('icon', 'fas fa-stethoscope'),
                'color': spec.get('color', '#3b82f6'),
                'is_active': spec.get('is_active', True)
            }
        return specialties
    except Exception as e:
        print(f"Error fetching specialties: {e}")
        return SPECIALTIES  # Fallback to hardcoded specialties


@app.route('/admin/specialties')
@admin_login_required
def admin_specialties():
    try:
        specialties = list(mongo.db.specialties.find().sort('name', 1))
        return render_template('Admin/specialties.html', specialties=specialties)
    except Exception as e:
        flash('Error loading specialties.', 'danger')
        return redirect(url_for('admin_dashboard'))


@app.route('/admin/specialties/add', methods=['GET', 'POST'])
@admin_login_required
def admin_add_specialty():
    if request.method == 'GET':
        return render_template('Admin/add_specialty.html')
    
    try:
        data = request.form
        
        # Validate required fields
        if not data.get('name') or not data.get('key'):
            flash('Specialty name and key are required.', 'danger')
            return redirect(url_for('admin_add_specialty'))
        
        # Check if key already exists
        existing = mongo.db.specialties.find_one({'key': data.get('key')})
        if existing:
            flash('Specialty key already exists.', 'danger')
            return redirect(url_for('admin_add_specialty'))
        
        # Create specialty document
        specialty_doc = {
            'key': data.get('key').lower().strip(),
            'name': data.get('name').strip(),
            'description': data.get('description', '').strip(),
            'icon': data.get('icon', 'fas fa-stethoscope'),
            'color': data.get('color', '#3b82f6'),
            'is_active': True,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        mongo.db.specialties.insert_one(specialty_doc)
        flash('Specialty added successfully!', 'success')
        return redirect(url_for('admin_specialties'))
        
    except Exception as e:
        print(f"Specialty creation failed: {e}")
        flash('Failed to add specialty.', 'danger')
        return redirect(url_for('admin_add_specialty'))


@app.route('/admin/specialties/<specialty_id>/edit', methods=['GET', 'POST'])
@admin_login_required
def admin_edit_specialty(specialty_id):
    try:
        specialty = mongo.db.specialties.find_one({'_id': ObjectId(specialty_id)})
        if not specialty:
            flash('Specialty not found.', 'danger')
            return redirect(url_for('admin_specialties'))
        
        if request.method == 'GET':
            return render_template('Admin/edit_specialty.html', specialty=specialty)
        
        # POST - update specialty
        data = request.form
        
        # Validate required fields
        if not data.get('name') or not data.get('key'):
            flash('Specialty name and key are required.', 'danger')
            return redirect(url_for('admin_edit_specialty', specialty_id=specialty_id))
        
        # Check if key already exists (excluding current specialty)
        existing = mongo.db.specialties.find_one({
            'key': data.get('key').lower().strip(),
            '_id': {'$ne': ObjectId(specialty_id)}
        })
        if existing:
            flash('Specialty key already exists.', 'danger')
            return redirect(url_for('admin_edit_specialty', specialty_id=specialty_id))
        
        # Update specialty document
        update_data = {
            'key': data.get('key').lower().strip(),
            'name': data.get('name').strip(),
            'description': data.get('description', '').strip(),
            'icon': data.get('icon', 'fas fa-stethoscope'),
            'color': data.get('color', '#3b82f6'),
            'is_active': 'is_active' in data,
            'updated_at': datetime.utcnow()
        }
        
        mongo.db.specialties.update_one(
            {'_id': ObjectId(specialty_id)},
            {'$set': update_data}
        )
        
        flash('Specialty updated successfully!', 'success')
        return redirect(url_for('admin_specialties'))
        
    except Exception as e:
        print(f"Specialty update failed: {e}")
        flash('Failed to update specialty.', 'danger')
        return redirect(url_for('admin_specialties'))


@app.route('/admin/specialties/<specialty_id>/delete', methods=['POST'])
@admin_login_required
def admin_delete_specialty(specialty_id):
    try:
        specialty = mongo.db.specialties.find_one({'_id': ObjectId(specialty_id)})
        if not specialty:
            flash('Specialty not found.', 'danger')
            return redirect(url_for('admin_specialties'))
        
        # Check if specialty is used in appointments
        appointment_count = mongo.db.appointments.count_documents({'specialty': specialty['key']})
        if appointment_count > 0:
            flash(f'Cannot delete specialty with {appointment_count} existing appointments.', 'danger')
            return redirect(url_for('admin_specialties'))
        
        # Check if specialty is used by doctors
        doctor_count = mongo.db.doctors.count_documents({'specialty': specialty['key']})
        if doctor_count > 0:
            flash(f'Cannot delete specialty used by {doctor_count} doctors.', 'danger')
            return redirect(url_for('admin_specialties'))
        
        mongo.db.specialties.delete_one({'_id': ObjectId(specialty_id)})
        flash('Specialty deleted successfully!', 'success')
        
    except Exception as e:
        print(f"Specialty deletion failed: {e}")
        flash('Failed to delete specialty.', 'danger')
    
    return redirect(url_for('admin_specialties'))


@app.route('/admin/specialties/<specialty_id>/toggle', methods=['POST'])
@admin_login_required
def admin_toggle_specialty(specialty_id):
    try:
        specialty = mongo.db.specialties.find_one({'_id': ObjectId(specialty_id)})
        if not specialty:
            return jsonify({'error': 'Specialty not found'}), 404
        
        new_status = not specialty.get('is_active', True)
        mongo.db.specialties.update_one(
            {'_id': ObjectId(specialty_id)},
            {'$set': {'is_active': new_status, 'updated_at': datetime.utcnow()}}
        )
        
        return jsonify({
            'success': True,
            'is_active': new_status,
            'message': f'Specialty {"activated" if new_status else "deactivated"} successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/specialty/<specialty_key>/doctors', methods=['GET'])
def get_doctors_for_specialty(specialty_key):
    try:
        # Get all active doctors for this specialty
        doctors = list(mongo.db.doctors.find({
            'specialty': specialty_key,
            'status': 'active'
        }, {
            '_id': 1,
            'full_name': 1,
            'email': 1,
            'qualification': 1,
            'experience_years': 1,
            'specialty': 1,
            'status': 1
        }))
        
        # Convert ObjectId to string for JSON serialization
        for doctor in doctors:
            doctor['_id'] = str(doctor['_id'])
        
        return jsonify({
            'success': True,
            'doctors': doctors
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/doctor/<doctor_id>/availability', methods=['GET'])
def get_doctor_availability_for_booking(doctor_id):
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not start_date or not end_date:
            return jsonify({'error': 'Start date and end date required'}), 400
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # 1) Get doctor's weekly schedule (only once)
        doctor = mongo.db.doctors.find_one({'_id': ObjectId(doctor_id)})
        if not doctor:
            return jsonify({'error': 'Doctor not found'}), 404

        # Preload weekly schedule for the doctor
        weekly_schedule = mongo.db.doctor_weekly_schedules.find_one({'doctor_email': doctor['email']})

        # 2) Preload daily overrides for the date range
        daily_overrides = {
            s['date'].date(): s
            for s in mongo.db.doctor_daily_schedules.find({
                'doctor_email': doctor['email'],
                'date': {'$gte': datetime.combine(start_dt.date(), datetime.min.time()),
                        '$lte': datetime.combine(end_dt.date(), datetime.min.time())}
            })
        }

        # 3) Preload all appointments for the date range
        appointments_by_day = {}
        
        # Find appointments that are active and should block a slot
        active_statuses = ['confirmed', 'pending', 'active']
        query = {
            'doctor_email': doctor['email'],
            'start_at': {'$gte': start_dt, '$lt': end_dt + timedelta(days=1)},
            'status': {'$in': active_statuses} # <-- FIX: Added status filter
        }
        
        for apt in mongo.db.appointments.find(query):
            day = apt['start_at'].date()
            appointments_by_day.setdefault(day, []).append(apt)

        # 4) Now loop through the date range and generate available slots in memory
        availability_data = []
        cur = start_dt
        while cur <= end_dt:
            # Get the daily override for the current day (if exists)
            daily_schedule = daily_overrides.get(cur.date())

            # Get appointments for the current day
            appointments = appointments_by_day.get(cur.date(), [])

            # Generate available time slots based on the schedule
            slots = generate_available_slots_for_date(cur, daily_schedule, weekly_schedule, appointments)

            # Append the result for the current date
            availability_data.append({
                'date': cur.strftime('%Y-%m-%d'),
                'available_slots': slots,
                'is_available': len(slots) > 0
            })

            cur += timedelta(days=1)

        return jsonify({
            'success': True,
            'doctor': {
                'id': str(doctor['_id']),
                'name': doctor['full_name'],
                'email': doctor['email'],
                'specialty': doctor['specialty']
            },
            'availability': availability_data
        })

    except Exception as e:
        print(f"Error in get_doctor_availability_for_booking: {e}")
        return jsonify({'error': str(e)}), 500


def generate_available_slots_for_date(date, daily_schedule, weekly_schedule, appointments):
    available_slots = []

    # If day explicitly marked unavailable via daily override
    if daily_schedule and daily_schedule.get('isAvailable') is False:
        return available_slots

    # Day name for weekly default
    day_name = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday'][date.weekday()]
    weekly_def = (weekly_schedule or {}).get('schedule', {}).get(day_name)

    # Booked starts (HH:MM)
    booked = {apt['start_at'].strftime('%H:%M') for apt in (appointments or [])}

    def add_minutes(hhmm, minutes):
        h, m = map(int, hhmm.split(':'))
        total = h*60 + m + minutes
        return f"{(total//60):02d}:{(total%60):02d}"

    # 1) Daily override: explicit list of slot starts like ["09:00", "09:30", ...]
    daily_slots = (daily_schedule or {}).get('time_slots') or []
    if isinstance(daily_slots, list) and daily_slots:
        duration = (weekly_def or {}).get('duration', 30)
        for s in sorted(daily_slots):
            if s not in booked:
                e = add_minutes(s, duration)
                available_slots.append({'start': s, 'end': e, 'formatted': f"{s} - {e}"})
        return available_slots

    # 2) Weekly default dict: {active, startTime, endTime, duration, breakStart, breakEnd}
    if isinstance(weekly_def, dict) and weekly_def.get('active', True):
        start = weekly_def.get('startTime')
        end = weekly_def.get('endTime')
        duration = weekly_def.get('duration', 30)
        break_start = weekly_def.get('breakStart')
        break_end = weekly_def.get('breakEnd')

        if start and end:
            cur = start
            while cur < end:
                # Respect break if defined
                in_break = bool(break_start and break_end and break_start <= cur < break_end)
                if not in_break and cur not in booked:
                    e = add_minutes(cur, duration)
                    if e <= end:
                        available_slots.append({'start': cur, 'end': e, 'formatted': f"{cur} - {e}"})
                cur = add_minutes(cur, duration)

    return available_slots


# Helper function to create a consistent conversation ID
def get_conversation_id(id1, id2):
    return '_'.join(sorted([str(id1), str(id2)]))

@app.route('/api/doctor/conversations')
@doctor_login_required
def get_doctor_conversations():
    doctor_id = session.get('doctor_id')
    
    pipeline = [
        {'$match': {'$or': [{'sender_id': doctor_id}, {'receiver_id': doctor_id}]}},
        {'$sort': {'timestamp': -1}},
        {'$group': {
            '_id': '$conversation_id',
            'last_message': {'$first': '$content'},
            'timestamp': {'$first': '$timestamp'},
            'sender_id': {'$first': '$sender_id'},
            'receiver_id': {'$first': '$receiver_id'}
        }},
        {'$sort': {'timestamp': -1}}
    ]
    conversations = list(mongo.db.conversations.aggregate(pipeline))

    for conv in conversations:
        other_user_id = conv['sender_id'] if conv['sender_id'] != doctor_id else conv['receiver_id']
        patient = mongo.db.users.find_one({'_id': ObjectId(other_user_id)})
        if patient:
            conv['other_user_name'] = patient['username']
            conv['other_user_id'] = str(patient['_id'])
        else:
            conv['other_user_name'] = 'Unknown Patient'
            conv['other_user_id'] = other_user_id
        # Convert ObjectId to string for JSON
        conv['_id'] = str(conv['_id'])
            
    return jsonify(conversations)

@app.route('/api/user/conversations')
@login_required
def get_user_conversations():
    user_id = current_user.id
    
    pipeline = [
        {'$match': {'$or': [{'sender_id': user_id}, {'receiver_id': user_id}]}},
        {'$sort': {'timestamp': -1}},
        {'$group': {
            '_id': '$conversation_id',
            'last_message': {'$first': '$content'},
            'timestamp': {'$first': '$timestamp'},
            'sender_id': {'$first': '$sender_id'},
            'receiver_id': {'$first': '$receiver_id'}
        }},
        {'$sort': {'timestamp': -1}}
    ]
    conversations = list(mongo.db.conversations.aggregate(pipeline))

    for conv in conversations:
        other_user_id = conv['sender_id'] if conv['sender_id'] != user_id else conv['receiver_id']
        doctor = mongo.db.doctors.find_one({'_id': ObjectId(other_user_id)})
        if doctor:
            conv['other_user_name'] = doctor['full_name']
            conv['other_user_id'] = str(doctor['_id'])
        else:
            conv['other_user_name'] = 'Unknown Doctor'
            conv['other_user_id'] = other_user_id
        # Convert ObjectId to string for JSON
        conv['_id'] = str(conv['_id'])

    return jsonify(conversations)

@app.route('/api/conversation/<conversation_id>')
def get_messages(conversation_id):

    if current_user.is_authenticated:
        user_id = current_user.id

    elif session.get('doctor_logged_in'):
        user_id = session.get('doctor_id')

    else:
        return jsonify({'error': 'Unauthorized'}), 401

    messages = list(mongo.db.conversations.find({
        'conversation_id': conversation_id
    }).sort('timestamp', 1))

    mongo.db.conversations.update_many(
        {'conversation_id': conversation_id, 'receiver_id': user_id, 'is_read': False},
        {'$set': {'is_read': True}}
    )

    for msg in messages:
        msg['_id'] = str(msg['_id'])

    return jsonify(messages)

@app.route('/api/conversation/send', methods=['POST'])
def send_message():
    data = request.get_json()
    receiver_id = data.get('receiver_id')
    content = data.get('content')

    if not receiver_id or not content:
        return jsonify({'error': 'Missing fields'}), 400

    if current_user.is_authenticated:
        sender_id = current_user.id
    elif session.get('doctor_logged_in'):
        sender_id = session.get('doctor_id')
    else:
        return jsonify({'error': 'Unauthorized'}), 401

    conversation_id = "_".join(sorted([sender_id, receiver_id]))

    message = {
        'conversation_id': conversation_id,
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'content': content,
        'timestamp': datetime.utcnow(),
        'is_read': False
    }

    mongo.db.conversations.insert_one(message)
    return jsonify({'success': True}), 201

@app.route('/conversations')
@login_required
def conversations_page():
    return render_template('dashboard/conversations.html')

@app.route('/chat/<conversation_id>')
@login_required
def chat_page(conversation_id):
    
    user_id = current_user.id
    ids = conversation_id.split('_')
    other_user_id = ids[0] if ids[0] != user_id else ids[1]
    
    other_user = mongo.db.doctors.find_one({'_id': ObjectId(other_user_id)})
    if not other_user:
        other_user = {'full_name': 'User'}

    return render_template('dashboard/chat.html', 
                           conversation_id=conversation_id, 
                           other_user_name=other_user['full_name'])

@app.route('/doctor/conversations')
@doctor_login_required
def doctor_conversations_page():
    return render_template('Doctor/conversations.html')

@app.route('/doctor/chat/<conversation_id>')
@doctor_login_required
def doctor_chat_page(conversation_id):
    doctor_id = session.get('doctor_id')
    ids = conversation_id.split('_')
    other_user_id = ids[0] if ids[0] != doctor_id else ids[1]

    other_user = mongo.db.users.find_one({'_id': ObjectId(other_user_id)})
    if not other_user:
        other_user = {'username': 'Unknown Patient'}

    return render_template('Doctor/chat.html', 
                           conversation_id=conversation_id, 
                           other_user_name=other_user['username'])

if __name__ == "__main__":
    app.run(debug=True)