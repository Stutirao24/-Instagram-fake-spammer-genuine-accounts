#  Instagram Fake Account Detector

A machine learning project to detect **fake vs genuine Instagram accounts** using data preprocessing, feature engineering, model training, explainability (SHAP), and deployment with **Streamlit**.

---

##  Project Workflow

### 🔹 Data Preprocessing
- Cleaned & transformed raw dataset  
- Added new engineered features:
  - log transforms (followers, follows, posts)  
  - ratio features (followers/follows)  
  - per-word description length  
  - boolean → integer conversions  

### 🔹 Model Training
- Performed **train-test split**  
- Built a pipeline with **Decision Tree Classifier** and **StandardScaler**  
- Achieved:  
  -  Accuracy: ~92%  
  -  ROC-AUC: 0.92  

### 🔹 Explainability
- Used **SHAP values** to identify most important features for classification  

###🔹 Deployment
- Built an interactive **Streamlit app**  
- Upload a CSV with Instagram account data  
- Get predictions: **fake_prediction** + **fake_probability**  
- Option to download results as CSV  

---

##  Project Structure
insta_fake_project/
│── app.py # Streamlit app
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── data/
│ ├── train.csv
│ ├── test.csv
│ ├── train_cleaned.csv
│ └── test_cleaned.csv
│── notebooks/
│ └── starter_instagram.ipynb
│── outputs/
│ ├── best_model.pkl
│ └── test_predictions.csv
│── src/ # (optional helper scripts)
│── .gitignore


---


## ⚙️ Installation

Clone this repo and create a virtual environment:

```bash
git clone https://github.com/YourUsername/instagram-fake-detector.git
cd instagram-fake-detector

# Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
