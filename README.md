# 📌 Flask Project Setup Guide

This project uses **Flask**, MongoDB, and Machine Learning libraries.  
Follow these steps to set up the project on your local machine.

Python version must be 3.13

---

## 🚀 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

## 🔧 2. Create Virtual Environment
Each developer should create their own virtual environment.

- **Windows (CMD)**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **Windows (PowerShell)**
  ```bash
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

- **Linux / macOS**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

## 📦 3. Install Dependencies
We are using a `requirements.txt` file to keep dependencies in sync.

```bash
pip install -r requirements.txt
```

---

## ⚙️ 4. Environment Variables
Create a `.env` file in the project root. Example:

```env
FLASK_ENV=development
SECRET_KEY=your_secret_key
MONGO_URI=mongodb://localhost:27017/your_database
```

---

## ▶️ 5. Run the Project
Make sure your virtual environment is activated, then run:

```bash
python app.py
```

By default, Flask runs on:  
👉 http://127.0.0.1:5000/

---

## 🧪 6. Updating Dependencies
If you add a new library, run:

```bash
pip freeze > requirements.txt
```

Commit and push the updated `requirements.txt` so everyone else can install it.

---

## 👥 Workflow for Collaboration
1. **Pull latest changes** before starting work:
   ```bash
   git pull origin main
   ```

2. **Make changes** locally inside your virtual environment.

3. **Test** your code.

4. **Commit and push** your changes:
   ```bash
   git add .
   git commit -m "Your message here"
   git push origin main
   ```

5. Teammates just need to pull:
   ```bash
   git pull origin main
   ```

---

## 📌 Notes
- Don’t commit your `venv/` folder or large files.  
- Use `.gitignore` to exclude:
  ```
  venv/
  __pycache__/
  .env
  *.pyc
  *.pkl
  *.h5
  ```
- Everyone maintains their **own virtual environment** locally.  
