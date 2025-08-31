from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, make_response
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
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
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

try:
    heart_model = joblib.load('model/heart_disease_model_6_features_v2.pkl')
    kidney_model = joblib.load('model/final_kidney_disease_model_train_10_features.pkl')
    retinopathy_model = load_model('model/Retinopathy-disease-v2.h5')
    chest_model = load_model('model/chest_xray_model.h5')

    heart_explainer = shap.TreeExplainer(heart_model)
    kidney_explainer = shap.TreeExplainer(kidney_model)

except FileNotFoundError:
    heart_model = None
    kidney_model = None
    retinopathy_model = None
    print("Warning: Model file not found. Prediction functionality will be disabled.")

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

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mongo = PyMongo(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'home'

# PDF Generation Class
class PDF(FPDF):
    THEME_PRIMARY = (59, 130, 246)  # Blue
    THEME_TEXT = (17, 24, 39)       # Slate-900
    THEME_MUTED = (75, 85, 99)      # Slate-600
    THEME_LIGHT = (249, 250, 251)   # Gray-50

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
    """Build a PDF report (bytes) for a given prediction document."""
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

            # Add doctor URL to response if the result is positive
            doctor_url = None
            if is_positive:
                if analysis_type == 'retinopathy':
                    doctor_url = url_for('eye_doctor')
                elif analysis_type == 'xray':
                    doctor_url = url_for('chest_doctor')
            user_response['doctor_url'] = doctor_url

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
        prediction = mongo.db.predictions.find__one({
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
    """
    Renders the heart_doctor.
    """
    return render_template('dashboard/heart_doctor.html')

@app.route('/eye_doctor')
def eye_doctor():
    """
    Renders the eye_doctor.
    """
    return render_template('dashboard/eye_doctor.html')

@app.route('/kidney_doctor')
def kidney_doctor():
    """
    Renders the kidney_doctor.
    """
    return render_template('dashboard/kidney_doctor.html')

@app.route('/chest_doctor')
def chest_doctor():
    """
    Renders the chest_doctor.
    """
    return render_template('dashboard/chest_doctor.html')

@app.route('/doctorsuggestion')
def doctor_suggestion():
    """
    Renders the doctorsuggestion.
    """
    return render_template('dashboard/doctorsuggestion.html')


if __name__ == "__main__":
    app.run(debug=True)

