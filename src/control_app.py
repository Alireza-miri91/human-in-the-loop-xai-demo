from flask import Flask, render_template_string, request, jsonify, send_file, session
import os, secrets, threading, socket, webbrowser, numpy as np, json, pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "runtime_data", "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "runtime_data", "cache"))
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("once", UserWarning)
import uuid
from datetime import datetime, timedelta

# Import the IGANN model
try:
    from .igann import IGANN
except ImportError:  # Allows running this file directly during local development.
    from igann import IGANN

DATA_PATH = os.environ.get('HIL_XAI_DATA_PATH', os.path.join(PROJECT_ROOT, 'data', 'day.csv'))
RUNTIME_DATA_DIR = os.environ.get('HIL_XAI_RUNTIME_DATA_DIR', os.path.join(PROJECT_ROOT, 'runtime_data'))
RESPONSES_DIR = os.path.join(RUNTIME_DATA_DIR, 'responses', 'control')
os.makedirs(RESPONSES_DIR, exist_ok=True)

# Load and preprocess the bike sharing data
def load_bike_data():
    df = pd.read_csv(DATA_PATH)
    
    # Select only numerical features for modeling (removed atemp)
    feature_cols = ['temp', 'hum', 'windspeed']
    target_col = 'cnt'  # total count of bike rentals
    
    # Denormalize the numerical features
    df["temp"] = df["temp"] * 41
    df["hum"] = df["hum"] * 100
    df["windspeed"] = df["windspeed"] * 67
    
    X = df[feature_cols].copy()
    y = df[target_col]
    
    return X, y

# Initialize data and model
X, y = load_bike_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the target variable
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

#y_train_scaled = y_train
#y_test_scaled = y_test

# Create base model
base_model = IGANN(
    task="regression",
    n_estimators=2,
    n_hid=10,
    boost_rate=0.1,
    init_reg=0.01,
    elm_scale=1,
    elm_alpha=0.01,
    act="elu",
    early_stopping=20,
    device="cpu",
    random_state=42,
    verbose=0
)

# Train base model
print("Training base model...")
print(f"X_train shape: {X_train.shape}")
print(f"y_train_scaled shape: {y_train_scaled.shape}")
print(f"X_train columns: {X_train.columns.tolist()}")

base_model.fit(X_train, y_train_scaled)
print(f"Base model trained with features: {base_model.numerical_cols}")
print(f"Base model coefficients: {base_model.linear_model.coef_}")
print(f"Base model intercept: {base_model.linear_model.intercept_}")
print("Base model training completed successfully")

# Get feature names and their effects
feature_names = X.columns.tolist()
pretty_names = {
    'temp': 'Temperature',
    'hum': 'Humidity',
    'windspeed': 'Wind Speed'
}

# Get base coefficients
print(f"Feature names: {feature_names}")
print(f"Model numerical_cols: {base_model.numerical_cols}")
base_coefficients = {}
for i, feat in enumerate(feature_names):
    try:
        idx = base_model.numerical_cols.index(feat)
        base_coefficients[feat] = float(base_model.linear_model.coef_[idx])
        print(f"Feature {feat}: coefficient {base_coefficients[feat]}")
    except ValueError as e:
        print(f"Error finding feature {feat} in model: {e}")
        base_coefficients[feat] = 0.0

# Create predictors list for the frontend (all features are numerical)
predictors = [
    dict(id=feat, name=pretty_names.get(feat, feat), 
         naive_coeff=base_coefficients[feat],
         sample_value=float(X[feat].mean()))
    for feat in feature_names
]

# Introduction page HTML
introduction_html = r"""
{% raw %}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bike Sharing – Introduction</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body{{font-family:ui-sans-serif,system-ui}}
  </style>
</head>
<body class="bg-gray-100">
  <div class="container mx-auto p-4 md:p-8 max-w-4xl">
    <div class="bg-white p-8 rounded-xl shadow-lg">
      <header class="text-center mb-8">
        <h1 class="text-4xl font-bold text-blue-800 mb-4">Welcome to Bike Sharing Demand Prediction</h1>
      </header>

      <div class="prose max-w-none">
        <p class="text-gray-700 mb-6 text-lg leading-relaxed">
          Welcome! You are invited to a short study about how people use a system for predicting daily bike-rental demand. You will (1) answer a brief pre-survey, (2) use a web app to review plots and make a few guesses about rental counts for sample cases, and (3) complete a short post-survey about your experience. The study takes about 10–15 minutes.
        </p>

        <p class="text-gray-700 mb-6 text-lg leading-relaxed">
          <strong>Your data.</strong> We collect your survey answers and anonymous interaction logs (e.g., choices you make in the app, time on page). We do not collect your name or email. Data are used for research only, stored securely, and kept no longer than needed for publication and verification.
        </p>

        <div class="text-center">
          <a href="#" onclick="window.location.href=window.location.pathname.replace(/\/[^/]*$/, '')+'/questionnaire'" class="inline-block bg-blue-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-blue-700 transition-colors text-lg">
            <i class="fas fa-clipboard-list mr-2"></i>Start Survey
          </a>
        </div>
      </div>
    </div>
  </div>
{% endraw %}
</body>
</html>
"""

# Questionnaire page HTML
questionnaire_html = r"""
{% raw %}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bike Sharing – Pre-Study Survey</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body{{font-family:ui-sans-serif,system-ui}}
  </style>
</head>
<body class="bg-gray-100">
  <div class="container mx-auto p-4 md:p-8 max-w-4xl">
    <div class="bg-white p-8 rounded-xl shadow-lg">
      <header class="text-center mb-8">
        <h1 class="text-3xl font-bold text-blue-800 mb-4">Pre-Study Survey</h1>
        <p class="text-gray-600">Please answer the following questions before using the model</p>
        

      </header>

      <form id="questionnaireForm" class="space-y-6">
        <!-- Demographics -->
        <div class="border-b pb-6">
          <h2 class="text-xl font-semibold text-gray-800 mb-4">Demographics</h2>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">Field <span class="text-red-500">*</span></label>
              <select name="field" class="w-full border border-gray-300 rounded-md px-3 py-2">
                <option value="">Select field</option>
                <option value="computer_science">Computer Science/Artificial Intelligence/Data Science</option>
                <option value="math_stats">Math-Stats</option>
                <option value="engineering">Engineering</option>
                <option value="natural_sciences">Natural Sciences</option>
                <option value="social_sciences">Social Sciences</option>
                <option value="business">Business</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">Education Level <span class="text-red-500">*</span></label>
              <select name="education" class="w-full border border-gray-300 rounded-md px-3 py-2">
                <option value="">Select education level</option>
                <option value="high_school">High School</option>
                <option value="bachelor">Bachelor's Degree</option>
                <option value="master">Master's Degree</option>
                <option value="phd">PhD</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Background and Experience -->
        <div class="border-b pb-6">
          <h2 class="text-xl font-semibold text-gray-800 mb-4">Background and Experience</h2>
          <div class="space-y-6">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">I'm familiar with machine learning. <span class="text-red-500">*</span></label>
              <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
                <span>Strongly disagree</span>
                <span>Strongly agree</span>
              </div>
              <div class="flex justify-between items-center">
                <label class="flex flex-col items-center">
                  <input type="radio" name="ml_familiarity" value="1" class="mr-1">
                  <span class="text-xs">1</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="ml_familiarity" value="2" class="mr-1">
                  <span class="text-xs">2</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="ml_familiarity" value="3" class="mr-1">
                  <span class="text-xs">3</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="ml_familiarity" value="4" class="mr-1">
                  <span class="text-xs">4</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="ml_familiarity" value="5" class="mr-1">
                  <span class="text-xs">5</span>
                </label>
              </div>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">I'm familiar with interpretable/explainable ML? <span class="text-red-500">*</span></label>
              <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
                <span>Strongly disagree</span>
                <span>Strongly agree</span>
              </div>
              <div class="flex justify-between items-center">
                <label class="flex flex-col items-center">
                  <input type="radio" name="interpretable_ml_familiarity" value="1" class="mr-1">
                  <span class="text-xs">1</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="interpretable_ml_familiarity" value="2" class="mr-1">
                  <span class="text-xs">2</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="interpretable_ml_familiarity" value="3" class="mr-1">
                  <span class="text-xs">3</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="interpretable_ml_familiarity" value="4" class="mr-1">
                  <span class="text-xs">4</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="interpretable_ml_familiarity" value="5" class="mr-1">
                  <span class="text-xs">5</span>
                </label>
              </div>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">I'm comfortable reading/understanding line charts. <span class="text-red-500">*</span></label>
              <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
                <span>Strongly disagree</span>
                <span>Strongly agree</span>
              </div>
              <div class="flex justify-between items-center">
                <label class="flex flex-col items-center">
                  <input type="radio" name="chart_comfort" value="1" class="mr-1">
                  <span class="text-xs">1</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="chart_comfort" value="2" class="mr-1">
                  <span class="text-xs">2</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="chart_comfort" value="3" class="mr-1">
                  <span class="text-xs">3</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="chart_comfort" value="4" class="mr-1">
                  <span class="text-xs">4</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="chart_comfort" value="5" class="mr-1">
                  <span class="text-xs">5</span>
                </label>
              </div>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">I'm familiar with bike-sharing systems? <span class="text-red-500">*</span></label>
              <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
                <span>Strongly disagree</span>
                <span>Strongly agree</span>
              </div>
              <div class="flex justify-between items-center">
                <label class="flex flex-col items-center">
                  <input type="radio" name="bike_sharing_familiarity" value="1" class="mr-1">
                  <span class="text-xs">1</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="bike_sharing_familiarity" value="2" class="mr-1">
                  <span class="text-xs">2</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="bike_sharing_familiarity" value="3" class="mr-1">
                  <span class="text-xs">3</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="bike_sharing_familiarity" value="4" class="mr-1">
                  <span class="text-xs">4</span>
                </label>
                <label class="flex flex-col items-center">
                  <input type="radio" name="bike_sharing_familiarity" value="5" class="mr-1">
                  <span class="text-xs">5</span>
                </label>
              </div>
            </div>


          </div>
        </div>

        <div class="text-center pt-6">
          <button type="submit" class="bg-blue-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-blue-700 transition-colors text-lg">
            <i class="fas fa-arrow-right mr-2"></i>Continue to Model
          </button>
        </div>
      </form>
    </div>
  </div>

  <script>
    // Track page time
    let pageStartTime = Date.now();
    
    // Send page time when leaving
    window.addEventListener('beforeunload', function() {
      const timeSpent = (Date.now() - pageStartTime) / 1000;
      fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/track-interaction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          event_type: 'page_time',
          page: 'questionnaire',
          time_spent: timeSpent
        })
      });
    });
    
    // Define required fields
    const requiredFields = {
      'field': 'Field',
      'education': 'Education Level',
      'ml_familiarity': 'Machine Learning Familiarity',
      'interpretable_ml_familiarity': 'Interpretable ML Familiarity',
      'chart_comfort': 'Chart Reading Comfort',
      'bike_sharing_familiarity': 'Bike Sharing Familiarity'
    };
    
    // Validation function
    function validateForm() {
      const form = document.getElementById('questionnaireForm');
      const formData = new FormData(form);
      const errors = [];
      
      // Check required fields
      for (const [fieldName, fieldLabel] of Object.entries(requiredFields)) {
        // For all fields (radio buttons, select elements)
        const value = formData.get(fieldName);
        console.log(`${fieldName}:`, value); // Debug logging
        if (!value || value === '') {
          errors.push(`${fieldLabel} is required`);
        }
      }
      
      console.log('Validation errors:', errors); // Debug logging
      return errors;
    }
    
    // Add visual indicators for required fields
    function addRequiredIndicators() {
      const requiredFieldNames = Object.keys(requiredFields);
      requiredFieldNames.forEach(fieldName => {
        const field = document.querySelector(`[name="${fieldName}"]`);
        if (field) {
          // Find the main question label by looking for the label that contains the question text
          // Go up to the parent div and find the first label that's not for an option
          const container = field.closest('div');
          const allLabels = container.querySelectorAll('label');
          
          // Find the main question label (the one that's not for radio/checkbox options)
          let mainLabel = null;
          for (let label of allLabels) {
            // Skip labels that are for options (they have flex class or are inside option containers)
            if (!label.classList.contains('flex') && 
                !label.classList.contains('items-center') &&
                !label.closest('.space-y-2')) {
              mainLabel = label;
              break;
            }
          }
          
          if (mainLabel && !mainLabel.innerHTML.includes('<span class="text-red-500">*</span>')) {
            mainLabel.innerHTML += ' <span class="text-red-500">*</span>';
          }
        }
      });
    }
    
    // Show validation errors
    function showErrors(errors) {
      // Remove existing error messages
      const existingErrors = document.querySelectorAll('.error-message');
      existingErrors.forEach(el => el.remove());
      
      // Create error container
      const errorContainer = document.createElement('div');
      errorContainer.className = 'bg-red-50 border border-red-200 rounded-md p-4 mb-4';
      errorContainer.innerHTML = `
        <h4 class="text-red-800 font-medium mb-2">Please fix the following errors:</h4>
        <ul class="list-disc list-inside text-red-700 space-y-1">
          ${errors.map(error => `<li>${error}</li>`).join('')}
        </ul>
      `;
      
      // Insert at the top of the form
      const form = document.getElementById('questionnaireForm');
      form.insertBefore(errorContainer, form.firstChild);
      
      // Scroll to top to show errors
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    document.getElementById('questionnaireForm').addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Validate form
      const errors = validateForm();
      if (errors.length > 0) {
        showErrors(errors);
        return;
      }
      
      // Collect form data
      const formData = new FormData(e.target);
      const data = {};
      for (let [key, value] of formData.entries()) {
        if (key === 'factors') {
          // Special handling for checkboxes - always create an array
          if (data[key]) {
            if (Array.isArray(data[key])) {
              data[key].push(value);
            } else {
              data[key] = [data[key], value];
            }
          } else {
            data[key] = [value];
          }
        } else {
          // For radio buttons and select elements
          if (data[key]) {
            if (Array.isArray(data[key])) {
              data[key].push(value);
            } else {
              data[key] = [data[key], value];
            }
          } else {
            data[key] = value;
          }
        }
      }
      
      // Store in sessionStorage for later use
      sessionStorage.setItem('questionnaireData', JSON.stringify(data));
      
      // Submit to server
      fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/submit-questionnaire`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      })
      .then(response => response.json())
      .then(result => {
        if (result.status === 'success') {
          // Redirect to main page
          window.location.href = window.location.pathname.replace(/\/[^/]*$/, '') + '/main';
        } else {
          alert('Error submitting questionnaire: ' + result.error);
        }
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Error submitting questionnaire. Please try again.');
      });
    });
    
    // Initialize required field indicators
    document.addEventListener('DOMContentLoaded', function() {
      addRequiredIndicators();
    });
  </script>
{% endraw %}
</body>
</html>
"""

# Thank you page HTML
thank_you_html = r"""
{% raw %}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bike Sharing – Thank You</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body{{font-family:ui-sans-serif,system-ui}}
  </style>
</head>
<body class="bg-gray-100">
  <div class="container mx-auto p-4 md:p-8 max-w-4xl">
    <div class="bg-white p-8 rounded-xl shadow-lg">
      <header class="text-center mb-8">
        <div class="text-6xl text-green-600 mb-4">
          <i class="fas fa-check-circle"></i>
        </div>
        <h1 class="text-4xl font-bold text-green-800 mb-4">Thank You!</h1>
        <p class="text-gray-600 text-lg">Your participation in this study has been completed successfully.</p>
      </header>

      <div class="prose max-w-none text-center">
        <p class="text-gray-700 mb-6 text-lg leading-relaxed">
          Thank you for taking the time to participate in our bike sharing demand prediction study. Your responses and interactions with the system will help us better understand how people use and interpret machine learning models.
        </p>

        <p class="text-gray-700 mb-6 text-lg leading-relaxed">
          <strong>Data Collection Complete.</strong> All your survey responses and interaction data have been securely recorded. This information will be used solely for research purposes and will help advance our understanding of human-AI interaction in predictive modeling.
        </p>

        <div class="bg-blue-50 p-6 rounded-lg border border-blue-200 mb-6">
          <h3 class="text-xl font-semibold text-blue-800 mb-3">Study Summary</h3>
          <p class="text-blue-700">
            You have successfully completed all parts of this research study, including the pre-study questionnaire, interaction with the bike sharing prediction model, and the post-study feedback survey.
          </p>
        </div>

        <p class="text-gray-600 text-sm">
          If you have any questions about this research, please contact the research team. You may now close this browser window.
        </p>
      </div>
    </div>
  </div>
{% endraw %}
</body>
</html>
"""

# Questionnaire continuation page HTML
questionnaire_continuation_html = r"""
{% raw %}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bike Sharing – Post-Study Survey</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body{{font-family:ui-sans-serif,system-ui}}
  </style>
</head>
<body class="bg-gray-100">
  <div class="container mx-auto p-4 md:p-8 max-w-4xl">
    <div class="bg-white p-8 rounded-xl shadow-lg">
      <header class="text-center mb-8">
        <h1 class="text-3xl font-bold text-blue-800 mb-4">Post-Study Survey</h1>
        <p class="text-gray-600">Please share your experience with the model</p>
      </header>

      <form id="continuationForm" class="space-y-6">
        <!-- Trust and Acceptance -->
        <div class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">I trusted the system's predictions. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="trusted_predictions" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="trusted_predictions" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="trusted_predictions" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="trusted_predictions" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="trusted_predictions" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">I felt confident relying on the system when making judgments. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="confident_relying" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="confident_relying" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="confident_relying" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="confident_relying" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="confident_relying" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">The system seemed consistent in how it handled similar cases. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="consistent_handling" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="consistent_handling" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="consistent_handling" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="consistent_handling" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="consistent_handling" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">I would use a system like this in the future. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="use_future" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="use_future" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="use_future" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="use_future" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="use_future" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">I was hesitant to rely on the system. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="hesitant_rely" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="hesitant_rely" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="hesitant_rely" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="hesitant_rely" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="hesitant_rely" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">I would recommend this system to others in my domain. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="recommend_domain" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="recommend_domain" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="recommend_domain" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="recommend_domain" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="recommend_domain" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Overall, I accept this system as a helpful decision-support tool. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="accept_decision_tool" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="accept_decision_tool" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="accept_decision_tool" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="accept_decision_tool" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="accept_decision_tool" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">In general, I trust computer tools that make and use predictions. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="trust_predictive_models" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="trust_predictive_models" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="trust_predictive_models" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="trust_predictive_models" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="trust_predictive_models" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">In my studies/work, I would use these computer tools. <span class="text-red-500">*</span></label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Strongly disagree</span>
              <span>Strongly agree</span>
            </div>
            <div class="flex justify-between items-center">
              <label class="flex flex-col items-center">
                <input type="radio" name="open_to_prediction_support" value="1" class="mr-1">
                <span class="text-xs">1</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="open_to_prediction_support" value="2" class="mr-1">
                <span class="text-xs">2</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="open_to_prediction_support" value="3" class="mr-1">
                <span class="text-xs">3</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="open_to_prediction_support" value="4" class="mr-1">
                <span class="text-xs">4</span>
              </label>
              <label class="flex flex-col items-center">
                <input type="radio" name="open_to_prediction_support" value="5" class="mr-1">
                <span class="text-xs">5</span>
              </label>
            </div>
          </div>
          <!-- Inserted open feedback field -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2" for="plus_feedback">Do you have any additional comments, suggestions, or feedback about your experience? (Optional)</label>
            <textarea id="plus_feedback" name="plus_feedback" rows="4" class="w-full border border-gray-300 rounded-md px-3 py-2" placeholder="Your feedback..."></textarea>
          </div>
        </div>
        <div class="text-center pt-6">
          <button type="submit" class="bg-green-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-green-700 transition-colors text-lg">
            <i class="fas fa-check mr-2"></i>Submit Feedback
          </button>
        </div>
      </form>
    </div>
  </div>

  <script>
    // Track page time
    let pageStartTime = Date.now();
    
    // Send page time when leaving
    window.addEventListener('beforeunload', function() {
      const timeSpent = (Date.now() - pageStartTime) / 1000;
      fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/track-interaction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          event_type: 'page_time',
          page: 'questionnaire_continuation',
          time_spent: timeSpent
        })
      });
    });
    
    // Define required fields for post-study questionnaire
    const requiredFields = {
      'trusted_predictions': 'Trust in System Predictions',
      'confident_relying': 'Confidence in Relying on System',
      'consistent_handling': 'System Consistency',
      'use_future': 'Future Use Intent',
      'hesitant_rely': 'Hesitation to Rely on System',
      'recommend_domain': 'Recommendation to Domain',
      'accept_decision_tool': 'Acceptance as Decision Tool',
      'trust_predictive_models': 'Trust in Predictive Models',
      'open_to_prediction_support': 'Openness to Prediction Support'
    };
    
    // Validation function
    function validateForm() {
      const form = document.getElementById('continuationForm');
      const formData = new FormData(form);
      const errors = [];
      
      // Check required fields
      for (const [fieldName, fieldLabel] of Object.entries(requiredFields)) {
        const value = formData.get(fieldName);
        if (!value || value === '') {
          errors.push(`${fieldLabel} is required`);
        }
      }
      
      return errors;
    }
    
    // Add visual indicators for required fields
    function addRequiredIndicators() {
      const requiredFieldNames = Object.keys(requiredFields);
      requiredFieldNames.forEach(fieldName => {
        const field = document.querySelector(`[name="${fieldName}"]`);
        if (field) {
          // Find the main question label by looking for the label that contains the question text
          // Go up to the parent div and find the first label that's not for an option
          const container = field.closest('div');
          const allLabels = container.querySelectorAll('label');
          
          // Find the main question label (the one that's not for radio/checkbox options)
          let mainLabel = null;
          for (let label of allLabels) {
            // Skip labels that are for options (they have flex class or are inside option containers)
            if (!label.classList.contains('flex') && 
                !label.classList.contains('items-center') &&
                !label.closest('.space-y-2')) {
              mainLabel = label;
              break;
            }
          }
          
          if (mainLabel && !mainLabel.innerHTML.includes('<span class="text-red-500">*</span>')) {
            mainLabel.innerHTML += ' <span class="text-red-500">*</span>';
          }
        }
      });
    }
    
    // Show validation errors
    function showErrors(errors) {
      // Remove existing error messages
      const existingErrors = document.querySelectorAll('.error-message');
      existingErrors.forEach(el => el.remove());
      
      // Create error container
      const errorContainer = document.createElement('div');
      errorContainer.className = 'bg-red-50 border border-red-200 rounded-md p-4 mb-4';
      errorContainer.innerHTML = `
        <h4 class="text-red-800 font-medium mb-2">Please fix the following errors:</h4>
        <ul class="list-disc list-inside text-red-700 space-y-1">
          ${errors.map(error => `<li>${error}</li>`).join('')}
        </ul>
      `;
      
      // Insert at the top of the form
      const form = document.getElementById('continuationForm');
      form.insertBefore(errorContainer, form.firstChild);
      
      // Scroll to top to show errors
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    document.getElementById('continuationForm').addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Validate form
      const errors = validateForm();
      if (errors.length > 0) {
        showErrors(errors);
        return;
      }
      
      // Collect form data
      const formData = new FormData(e.target);
      const data = {};
      for (let [key, value] of formData.entries()) {
        if (key === 'factors') {
          // Special handling for checkboxes - always create an array
          if (data[key]) {
            if (Array.isArray(data[key])) {
              data[key].push(value);
            } else {
              data[key] = [data[key], value];
            }
          } else {
            data[key] = [value];
          }
        } else {
          // For radio buttons and select elements
          if (data[key]) {
            if (Array.isArray(data[key])) {
              data[key].push(value);
            } else {
              data[key] = [data[key], value];
            }
          } else {
            data[key] = value;
          }
        }
      }
      
      // Add plus_feedback from textarea
      const plusFeedback = document.getElementById('plus_feedback')?.value || '';
      data['plus_feedback'] = plusFeedback;
      
      // Store in sessionStorage
      sessionStorage.setItem('continuationData', JSON.stringify(data));
      
      // Submit to server
      fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/submit-continuation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      })
      .then(response => response.json())
      .then(result => {
        if (result.status === 'success') {
          // Redirect to thank you page
          window.location.href = window.location.pathname.replace(/\/[^/]*$/, '') + '/thank-you';
        } else {
          alert('Error submitting feedback: ' + result.error);
        }
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Error submitting feedback. Please try again.');
      });
    });
    
    // Initialize required field indicators
    document.addEventListener('DOMContentLoaded', function() {
      addRequiredIndicators();
    });
  </script>
{% endraw %}
</body>
</html>
"""

html = r"""
{% raw %}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bike Sharing Demand Prediction</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body{{font-family:ui-sans-serif,system-ui}}
    .prediction-input:valid {{ border-color: #10b981; background-color: #f0fdf4; }}
    .prediction-input:invalid {{ border-color: #ef4444; background-color: #fef2f2; }}
    .prediction-input:placeholder-shown {{ border-color: #d1d5db; background-color: white; }}
  </style>
</head>
<body class="bg-gray-100">

<div class="container mx-auto p-4 md:p-8 max-w-7xl">
  <header class="text-center mb-8">
    <h1 class="text-3xl font-bold text-blue-800">Bike Sharing Demand Prediction</h1>
    <p class="text-gray-600 mt-2"> Machine Learning Model</p>
    <div class="mt-2 text-xs text-gray-500">
      <span id="subjectInfo">Loading subject information...</span>
    </div>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
    <!-- CENTER PANEL - Results -->
    <div class="lg:col-span-3">
      <div class="bg-white p-6 rounded-xl shadow-lg">
        <h2 class="text-xl font-semibold mb-4 border-b pb-3 text-gray-800">Model Results</h2>

        <!-- Shape Functions Explanation -->
        <div class="mb-6">
          <div class="bg-green-50 p-4 rounded-lg border border-green-200">
            <h4 class="font-medium text-green-800 mb-2">
              <i class="fas fa-chart-line mr-2"></i>Understanding Shape Functions
            </h4>
            <p class="text-base text-green-700">
                The following shape functions show how each feature pushes the prediction up or down relative to an average day (baseline). For example, a value of –1500 at temperature = 5°C means the model predicts 1,500 fewer rentals than average. For any case, look at each feature's value on its curve and add them all to the baseline to get the final prediction (positives increase rentals, negatives decrease them).
            </p>
          </div>
        </div>

        <!-- Plots Section -->
        <div class="mb-6">
          <h3 class="font-medium text-gray-700 mb-3">Model Visualizations</h3>
          <div class="space-y-6">
            <div>
              <h4 class="font-medium text-gray-600 mb-2">Shape Functions Comparison</h4>
              <div id="shapePlot" class="bg-gray-50 rounded-lg p-4 min-h-96 flex items-center justify-center">
                <p class="text-gray-500">Loading model shape functions...</p>
              </div>
              <div class="mt-2 text-xs text-gray-500">
              </div>
            </div>
          </div>
        </div>

        <!-- Prediction Section -->
        <div class="mb-6">
          <h3 class="font-medium text-gray-700 mb-3">Model Predictions</h3>
          <div class="space-y-4">
            
            <div class="bg-blue-50 p-4 rounded-lg">
              <h4 class="font-medium text-blue-800 mb-3">Sample Predictions</h4>
              <p class="text-base text-blue-700 mb-4">Now imagine you are the resource manager for a bike-sharing company planning tomorrow’s fleet. For each case, enter how many more or fewer bikes than an average day you would stage. Enter your predictions for bike sharing demand compared to an average day for each weather condition (Note: negative values are possible), then see how the model compares:</p>
              
              <div class="grid grid-cols-1 gap-4">
                <!-- User Predictions Section -->
                <div id="userPredictionsSection">
                  <h5 class="font-medium text-gray-700 mb-2">Your Predictions</h5>
                  <p class="text-sm text-gray-600 mb-3">Enter your best guess for bike sharing demand compared to an average day for each weather condition (Note: negative values are possible):</p>
                  <div id="userPredictions" class="space-y-2 text-sm">
                    <div class="flex justify-between items-center p-2 bg-white rounded border">
                      <span class="text-gray-600">Loading sample conditions...</span>
                    </div>
                  </div>
                  <div id="submitPredictionsSection" class="mt-4 hidden">
                    <div class="flex items-center justify-between mb-3">
                      <span id="predictionProgress" class="text-sm text-gray-600">0 of 0 predictions filled</span>
                      <span class="text-xs text-gray-500">All fields required</span>
                    </div>
                    <button id="submitPredictionsBtn" class="bg-green-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-green-700 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed" disabled>
                      <i class="fas fa-check mr-2"></i>Submit My Predictions
                    </button>
                  </div>
                </div>
                
                <!-- Model Predictions Section (Hidden Initially) -->
                <div id="constrainedPredictionsSection" class="hidden">
                  <h5 class="font-medium text-gray-700 mb-2">Model Predictions</h5>
                  <div id="predictionSummary" class="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <h6 class="font-medium text-blue-800 mb-2">Prediction Summary</h6>
                    <div id="summaryContent" class="text-sm text-blue-700">
                      <!-- Summary will be populated here -->
                    </div>
                  </div>
                  <div id="constrainedPredictions" class="space-y-2 text-sm">
                  </div>
                  <div class="mt-4">
                    <button id="resetPredictionsBtn" class="bg-gray-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors text-sm">
                      <i class="fas fa-undo mr-2"></i>Go Back & Change My Predictions
                    </button>
                  </div>
                </div>
              </div>
              

              
              <!-- Acceptability Intention Section -->
              <div class="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                <h5 class="font-medium text-yellow-800 mb-3">User Feedback</h5>
                <p class="text-sm text-yellow-700 mb-4">Please rate your agreement with the following statement about using this system:</p>
                
                <div class="space-y-3">
                  <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">If I had to plan tomorrow's bikes, I would use this system's predictions as my default. <span class="text-red-500">*</span></label>
                    <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
                      <span>Strongly disagree</span>
                      <span>Strongly agree</span>
                    </div>
                    <div class="flex justify-between items-center">
                      <label class="flex flex-col items-center">
                        <input type="radio" name="acceptability_intention" value="1" class="mr-1">
                        <span class="text-xs">1</span>
                      </label>
                      <label class="flex flex-col items-center">
                        <input type="radio" name="acceptability_intention" value="2" class="mr-1">
                        <span class="text-xs">2</span>
                      </label>
                      <label class="flex flex-col items-center">
                        <input type="radio" name="acceptability_intention" value="3" class="mr-1">
                        <span class="text-xs">3</span>
                      </label>
                      <label class="flex flex-col items-center">
                        <input type="radio" name="acceptability_intention" value="4" class="mr-1">
                        <span class="text-xs">4</span>
                      </label>
                      <label class="flex flex-col items-center">
                        <input type="radio" name="acceptability_intention" value="5" class="mr-1">
                        <span class="text-xs">5</span>
                      </label>
                    </div>
                  </div>
                  
                  <div class="mt-4">
                    <button id="saveFeedbackBtn" class="bg-yellow-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-yellow-700 transition-colors text-sm">
                      <i class="fas fa-save mr-2"></i>Save Feedback</button>
                    <span id="feedbackStatus" class="ml-3 text-sm"></span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Navigation to post-study questionnaire -->
    <div class="mt-8 text-center">
      <button id="continueBtn"
              class="inline-block bg-green-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-green-700 transition-colors text-lg disabled:opacity-50 disabled:cursor-not-allowed"
              disabled>
        <i class="fas fa-comments mr-2"></i>Continue to Post-Study Questionnaire
      </button>
      <div id="continueHint" class="mt-2 text-sm text-gray-500">
        To continue, please submit your guesses and answer the question above.
      </div>
    </div>
  </div>
</div>

<script>
{% endraw %}
  const predictors = {{ predictors | tojson | safe }};
{% raw %}

/* -------- state -------- */
let currentModel = null;
let currentMetrics = null;

/* -------- helpers -------- */
// No renderMono function needed for control group





function displayUserPredictionInputs(predictions) {
  const container = document.getElementById('userPredictions');
  if (!container) return;
  
  container.innerHTML = predictions.map((pred, index) => {
    const temp = pred.temperature;
    const humidity = pred.humidity;
    const wind = pred.windspeed;
    
    return `
      <div class="flex justify-between items-center p-3 bg-white rounded border hover:bg-gray-50">
        <div class="flex-1">
          <div class="font-medium text-gray-800 mb-2">Weather Conditions:</div>
          <div class="space-y-2 text-sm">
            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Temperature:</span>
              <span class="text-gray-800 ml-2">${temp}°C</span>
            </div>
            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Humidity:</span>
              <span class="text-gray-800 ml-2">${humidity}%</span>
            </div>
            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Wind Speed:</span>
              <span class="text-gray-800 ml-2">${wind} km/h</span>
            </div>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <input type="number" 
                 id="userPrediction_${index}" 
                 class="w-20 px-2 py-1 border border-gray-300 rounded text-center text-sm prediction-input" 
                 placeholder="Guess"
                 min="-10000" 
                 max="10000"
                 data-prediction-index="${index}">
          <span class="text-xs text-gray-500">bikes per day more/fewer than an average day</span>
        </div>
      </div>
    `;
  }).join('');
  
  // Show submit button
  document.getElementById('submitPredictionsSection').classList.remove('hidden');
  
  // Update progress display
  updatePredictionProgress();
  
  // Add event listeners to all prediction inputs
  const predictionInputs = document.querySelectorAll('[id^="userPrediction_"]');
  predictionInputs.forEach(input => {
    input.addEventListener('input', updatePredictionProgress);
  });
}

function updatePredictionProgress() {
  const predictionInputs = document.querySelectorAll('[id^="userPrediction_"]');
  const totalPredictions = predictionInputs.length;
  let filledPredictions = 0;
  
  predictionInputs.forEach(input => {
    if (input.value && parseInt(input.value) >= -10000 && parseInt(input.value) <= 10000) {
      filledPredictions++;
    }
  });
  
  const progressEl = document.getElementById('predictionProgress');
  const submitBtn = document.getElementById('submitPredictionsBtn');
  
  if (progressEl) {
    progressEl.textContent = `${filledPredictions} of ${totalPredictions} predictions filled`;
  }
  
  if (submitBtn) {
    submitBtn.disabled = filledPredictions < totalPredictions;
  }
}



function displayPredictions(predictions, containerId) {
  console.log('displayPredictions called with:', predictions, 'containerId:', containerId);
  const container = document.getElementById(containerId);
  console.log('Found container:', container);
  if (!container) {
    console.error('Container not found for ID:', containerId);
    return;
  }
  
  // Calculate summary statistics
  let totalUserPredictions = 0;
  let totalModelPredictions = 0;
  let correctPredictions = 0;
  let closePredictions = 0;
  let totalPredictions = 0;
  
  predictions.forEach((pred, index) => {
    const userPredictionEl = document.getElementById(`userPrediction_${index}`);
    const userPrediction = userPredictionEl ? parseInt(userPredictionEl.value) || 0 : 0;
    const modelPrediction = Math.round(pred.prediction);
    
    if (userPrediction !== 0) {  // Changed from > 0 to !== 0 to include negative values
      totalUserPredictions += userPrediction;
      totalModelPredictions += modelPrediction;
      totalPredictions++;
      
      const diff = Math.abs(userPrediction - modelPrediction);
      const percentageDiff = (diff / modelPrediction) * 100;
      
      if (percentageDiff <= 10) {
        correctPredictions++;
      } else if (percentageDiff <= 25) {
        closePredictions++;
      }
    }
  });
  
  // Update summary
  if (totalPredictions > 0) {
    const summaryContent = document.getElementById('summaryContent');
    if (summaryContent) {
      const avgUserPrediction = Math.round(totalUserPredictions / totalPredictions);
      const avgModelPrediction = Math.round(totalModelPredictions / totalPredictions);
      const accuracy = Math.round((correctPredictions / totalPredictions) * 100);
      const closeAccuracy = Math.round(((correctPredictions + closePredictions) / totalPredictions) * 100);
      
      summaryContent.innerHTML = `
        <div class="grid grid-cols-2 gap-4 text-xs">
          <div>
            <strong>Your Average:</strong> ${avgUserPrediction.toLocaleString()} bikes per day more/fewer than an average day<br>
            <strong>Model Average:</strong> ${avgModelPrediction.toLocaleString()} bikes per day more/fewer than an average day
          </div>
          <div>
            <strong>Very Close (≤10%):</strong> ${correctPredictions}/${totalPredictions} (${accuracy}%)<br>
            <strong>Close (≤25%):</strong> ${correctPredictions + closePredictions}/${totalPredictions} (${closeAccuracy}%)
          </div>
        </div>
      `;
    }
  }
  
  console.log('Processing predictions:', predictions);
  container.innerHTML = predictions.map((pred, index) => {
    console.log('Processing prediction:', pred, 'index:', index);
    const temp = pred.temperature;
    const humidity = pred.humidity;
    const wind = pred.windspeed;
    const prediction = pred.prediction;
    const demand = Math.round(prediction);
    console.log('Extracted values:', { temp, humidity, wind, prediction, demand });
    
    // Get user prediction if available
    const userPredictionEl = document.getElementById(`userPrediction_${index}`);
    const userPrediction = userPredictionEl ? parseInt(userPredictionEl.value) || 0 : 0;
    
    // Color coding based on demand level
    let demandClass = 'text-gray-600';
    if (demand > 0) {
      demandClass = 'text-green-600';
    } else if (demand < 0) {
      demandClass = 'text-red-600';
    }
    
    // Compare user vs model prediction
    let comparisonClass = '';
    if (userPrediction !== 0) {  // Changed from > 0 to !== 0 to include negative values
      const diff = Math.abs(userPrediction - demand);
      const percentageDiff = (diff / Math.abs(demand)) * 100;
      if (percentageDiff <= 25) {
        comparisonClass = 'text-green-600';
      } else {
        comparisonClass = 'text-red-600';
      }
    }
    
    const html = `
      <div class="flex justify-between items-center p-3 bg-white rounded border hover:bg-gray-50">
        <div class="flex-1">
          <div class="font-medium text-gray-800 mb-2">Weather Conditions:</div>
          <div class="space-y-2 text-sm">
            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Temperature:</span>
              <span class="text-gray-800 ml-2">${temp}°C</span>
            </div>

            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Humidity:</span>
              <span class="text-gray-800 ml-2">${humidity}%</span>
            </div>
            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Wind Speed:</span>
              <span class="text-gray-800 ml-2">${wind} km/h</span>
            </div>
          </div>
        </div>
        <div class="text-right">
          <div class="font-bold ${demandClass} text-lg">${demand.toLocaleString()}</div>
          <div class="text-xs text-gray-500">
            bikes per day more/fewer than an average day
          </div>
          ${userPrediction !== 0 ? `
            <div class="mt-1 text-xs ${comparisonClass}">
              Your guess: ${userPrediction.toLocaleString()}
            </div>
          ` : ''}
        </div>
      </div>
    `;
    console.log('Generated HTML for prediction', index, ':', html);
    return html;
  }).join('');
  
  console.log('Final HTML for container:', container.innerHTML);
}



function savePredictionFeedback() {
  const acceptabilityIntention = document.querySelector('input[name="acceptability_intention"]:checked');
  
  if (!acceptabilityIntention) {
    document.getElementById('feedbackStatus').innerHTML = '<span class="text-red-600">Please rate your agreement with the statement</span>';
    return;
  }
  
  const feedback = {
    acceptability_intention: acceptabilityIntention.value,
    timestamp: new Date().toISOString()
  };
  
  // Show loading state
  document.getElementById('saveFeedbackBtn').innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Saving...';
  document.getElementById('saveFeedbackBtn').disabled = true;
  
  fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/save-prediction-feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(feedback)
  })
  .then(response => response.json())
  .then(result => {
    if (result.status === 'success') {
      document.getElementById('feedbackStatus').innerHTML = '<span class="text-green-600">✓ Feedback saved successfully!</span>';
      document.getElementById('saveFeedbackBtn').innerHTML = '<i class="fas fa-check mr-2"></i>Saved';
    } else {
      document.getElementById('feedbackStatus').innerHTML = '<span class="text-red-600">Error saving feedback</span>';
      document.getElementById('saveFeedbackBtn').innerHTML = '<i class="fas fa-save mr-2"></i>Save Feedback';
      document.getElementById('saveFeedbackBtn').disabled = false;
    }
  })
  .catch(error => {
    console.error('Error:', error);
    document.getElementById('feedbackStatus').innerHTML = '<span class="text-red-600">Error saving feedback</span>';
    document.getElementById('saveFeedbackBtn').innerHTML = '<i class="fas fa-save mr-2"></i>Save Feedback';
    document.getElementById('saveFeedbackBtn').disabled = false;
  });
}

// === DOM SAFETY HELPERS ===
function $(id){ return document.getElementById(id); }
function on(id, ev, fn){ const el = document.getElementById(id); if (el) el.addEventListener(ev, fn); }

/* -------- DOM refs -------- */
const shapePlotEl = document.getElementById('shapePlot');
const constrainedPredictionsSection = document.getElementById('constrainedPredictionsSection');
const constrainedPredictionsEl = document.getElementById('constrainedPredictions');

/* -------- init -------- */
function initializeApp() {
  console.log('Initializing app...');
  
  // Load and display subject information
  fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/get-subject-info`)
    .then(response => response.json())
    .then(data => {
      const subjectInfo = document.getElementById("subjectInfo");
      if (subjectInfo) {
        subjectInfo.textContent = `Subject ID: ${data.subject_id}`;
        console.log('Subject info loaded');
      }
    })
    .catch(error => {
      console.error('Error loading subject info:', error);
    });

  // Load base model shape functions by default
  fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/base-model-shapes`)
    .then(response => response.json())
    .then(data => {
      const shapePlot = document.getElementById("shapePlot");
      if (data.error) {
        if (shapePlot) shapePlot.innerHTML = '<p class="text-red-500">Error loading base model shape functions</p>';
        return;
      }
      if (data.shape_plot && shapePlot) {
        shapePlot.innerHTML = `<img src="data:image/png;base64,${data.shape_plot}" style="width: 900px; max-width: 100%; height: auto;">`;
        console.log('Shape functions loaded');
      }
    })
    .catch(error => {
      console.error('Error loading shape functions:', error);
      const shapePlot = document.getElementById("shapePlot");
      if (shapePlot) shapePlot.innerHTML = '<p class="text-red-500">Error loading base model shape functions</p>';
    });

  // Load base model sample predictions and display guess box
  fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/base-model-predictions`)
    .then(response => response.json())
    .then(data => {
      console.log('Received predictions data:', data);
      if (data.predictions) {
        window.currentPredictions = data.predictions;
        displayUserPredictionInputs(data.predictions);
        console.log('Predictions loaded and displayed');
        
        // Attach event listeners after elements are created
        setTimeout(() => {
          on('submitPredictionsBtn', 'click', submitUserPredictions);
          on('resetPredictionsBtn', 'click', resetUserPredictions);
          on('saveFeedbackBtn', 'click', savePredictionFeedback);
          console.log('Event listeners attached');
        }, 100);
      } else {
        console.error('No predictions data received:', data);
        const userPredictions = document.getElementById("userPredictions");
        if (userPredictions) {
          userPredictions.innerHTML = '<div class="p-2 bg-red-50 border border-red-200 rounded"><span class="text-red-600">Error: No predictions data received</span></div>';
        }
      }
    })
    .catch(error => {
      console.error('Error loading predictions:', error);
      const userPredictions = document.getElementById("userPredictions");
      if (userPredictions) {
        userPredictions.innerHTML = `<div class="p-2 bg-red-50 border border-red-200 rounded"><span class="text-red-600">Error loading predictions: ${error.message}</span></div>`;
      }
    });

  // Also attach event listeners on DOMContentLoaded as backup
  document.addEventListener('DOMContentLoaded', function() {
    on('submitPredictionsBtn', 'click', submitUserPredictions);
    on('resetPredictionsBtn', 'click', resetUserPredictions);
    on('saveFeedbackBtn', 'click', savePredictionFeedback);
  });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}

/* -------- events -------- */
// No constraints handling needed for control group

// Track page time
let pageStartTime = Date.now();

// Send page time when leaving
window.addEventListener('beforeunload', function() {
  const timeSpent = (Date.now() - pageStartTime) / 1000;
  fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/track-interaction`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      event_type: 'page_time',
      page: 'main',
      time_spent: timeSpent
    })
  });
});

// ... existing code ...
// Remove all retrain/updateBtn logic and references
// Remove any code that posts to /train or references constraints
// Remove renderMono and monoWrap logic (not needed for control)
// ... existing code ...

// Update displayUserPredictionInputs to use neutral message
function displayUserPredictionInputs(predictions) {
  console.log('displayUserPredictionInputs called with:', predictions);
  const container = document.getElementById("userPredictions");
  console.log('Found userPredictions container:', container);
  if (!container) {
    console.error('userPredictions container not found!');
    return;
  }
  container.innerHTML = predictions.map((pred, index) => {
    const temp = pred.temperature;
    const humidity = pred.humidity;
    const wind = pred.windspeed;
    return `
      <div class="flex justify-between items-center p-3 bg-white rounded border hover:bg-gray-50">
        <div class="flex-1">
          <div class="font-medium text-gray-800 mb-2">Weather Conditions:</div>
          <div class="space-y-2 text-sm">
            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Temperature:</span>
              <span class="text-gray-800 ml-2">${temp}°C</span>
            </div>
            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Humidity:</span>
              <span class="text-gray-800 ml-2">${humidity}%</span>
            </div>
            <div class="flex items-center">
              <span class="font-medium text-blue-600 w-24">Wind Speed:</span>
              <span class="text-gray-800 ml-2">${wind} km/h</span>
            </div>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <input type="number" 
                 id="userPrediction_${index}" 
                 class="w-20 px-2 py-1 border border-gray-300 rounded text-center text-sm prediction-input" 
                 placeholder="Guess"
                 min="0" 
                 max="10000"
                 data-prediction-index="${index}">
          <span class="text-xs text-gray-500">bikes per day more/fewer than an average day</span>
        </div>
      </div>
    `;
  }).join('');
  // Show submit button
  const submitPredictionsSection = document.getElementById('submitPredictionsSection');
  if (submitPredictionsSection) submitPredictionsSection.classList.remove('hidden');
  // Update progress display
  updatePredictionProgress();
  // Add event listeners to all prediction inputs
  const predictionInputs = document.querySelectorAll('[id^="userPrediction_"]');
  predictionInputs.forEach(input => {
    input.addEventListener('input', updatePredictionProgress);
  });
  // Update copy
  const userPredictionsSection = document.getElementById('userPredictionsSection');
  if (userPredictionsSection) {
    const msg = userPredictionsSection.querySelector('span.text-gray-600');
    if (msg) msg.textContent = "Enter your guesses for the conditions below, then click Submit to see the model's predictions.";
  }
}

function submitUserPredictions() {
  // Get all user predictions
  const userPredictions = [];
  const predictionInputs = document.querySelectorAll('[id^="userPrediction_"]');
  let allFilled = true;
  predictionInputs.forEach((input, index) => {
    const value = parseInt(input.value) || 0;
    if (value < -10000 || value > 10000) {
      allFilled = false;
    }
    userPredictions.push(value);
  });
  if (!allFilled) {
    alert('Please enter predictions for all sample conditions before submitting.');
    return;
  }
  // Hide user predictions section
  const userPredictionsSection = document.getElementById('userPredictionsSection');
  if (userPredictionsSection) userPredictionsSection.classList.add('hidden');
  // Show model predictions section
  const constrainedPredictionsSection = document.getElementById('constrainedPredictionsSection');
  if (constrainedPredictionsSection) constrainedPredictionsSection.classList.remove('hidden');
  // Show base model predictions (reuse window.currentPredictions)
  console.log('window.currentPredictions:', window.currentPredictions);
  if (window.currentPredictions) {
    console.log('Calling displayPredictions with:', window.currentPredictions);
    displayPredictions(window.currentPredictions, 'constrainedPredictions');
  } else {
    console.error('No currentPredictions found!');
  }
  // Analytics hook (optional)
  fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/track-interaction`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_type: 'control_submit',
      user_predictions: userPredictions
    })
  });
  // Gating: mark predictions as submitted and update continue button
  hasSubmittedPredictions = true;
  updateContinueAvailability();
}

function resetUserPredictions() {
  // Show user predictions section again
  const userPredictionsSection = document.getElementById('userPredictionsSection');
  if (userPredictionsSection) userPredictionsSection.classList.remove('hidden');
  // Hide model predictions section
  const constrainedPredictionsSection = document.getElementById('constrainedPredictionsSection');
  if (constrainedPredictionsSection) constrainedPredictionsSection.classList.add('hidden');
  // Clear all user input values and re-render
  if (window.currentPredictions) {
    displayUserPredictionInputs(window.currentPredictions);
  }
  // Analytics hook (optional)
  fetch(`${window.location.pathname.replace(/\/[^/]*$/, '')}/track-interaction`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_type: 'control_view_results',
    })
  });
}
// ... existing code ...
// Remove any remaining references to constraints, updateBtn, retrain, or /train endpoints.

// --- gating state & logic (Control) ---
let hasSubmittedPredictions = false;

function updateContinueAvailability() {
  const btn = document.getElementById("continueBtn");
  const hint = document.getElementById("continueHint");
  if (!btn) return;

  const answered = !!document.querySelector('input[name="acceptability_intention"]:checked');

  if (hasSubmittedPredictions && answered) {
    btn.disabled = false;
    if (hint) hint.textContent = 'All set — you can continue.';
  } else {
    btn.disabled = true;
    if (hint) {
      const missing = [];
      if (!hasSubmittedPredictions) missing.push('submit your guesses');
      if (!answered) missing.push('answer the question');
      hint.textContent = `To continue, please ${missing.join(' and ')}.`;
    }
  }
}

// react to the bottom question selection
document.addEventListener('change', (e) => {
  if (e.target && e.target.name === 'acceptability_intention') {
    updateContinueAvailability();
  }
});

// handle continue click
on('continueBtn', 'click', () => {
  window.location.href = window.location.pathname.replace(/\/[^/]*$/, '') + '/questionnaire-continuation';
});
</script>
</body>
</html>
{% endraw %}
"""

def create_numerical_shape_plot(model, base_model=None):
    """Create shape functions plot for numerical features only"""
    numerical_features = model.numerical_cols if hasattr(model, 'numerical_cols') else ['temp', 'hum', 'windspeed']
    try:
        shape_functions = model.get_shape_functions_as_dict()
        numerical_shapes = {k: v for k, v in shape_functions.items() if k in numerical_features}
        all_y_values = []
        
        # Collect y values in scaled units (no denormalization)
        for shape_func in numerical_shapes.values():
            if shape_func['datatype'] == 'numerical' and 'y' in shape_func and len(shape_func['y']) > 0:
                all_y_values.extend(shape_func['y'])
                
        if all_y_values:
            global_y_min = min(all_y_values)
            global_y_max = max(all_y_values)
            y_padding = (global_y_max - global_y_min) * 0.1
            
            # Ensure 0 is included in the y-axis range
            y_min = min(global_y_min - y_padding, 0)
            y_max = max(global_y_max + y_padding, 0)
        else:
            # Default values if no data (scaled units) - ensure 0 is included
            y_min, y_max = -0.8, 0.8
            
        n_features = len(numerical_shapes)
        if n_features == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No numerical features found', ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            return fig
        
        # Create subplots with 1 column: shape functions only
        n_cols = 1
        n_rows = n_features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        
        # Ensure axes is always a 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (feature_name, shape_func) in enumerate(numerical_shapes.items()):
            row = i
            
            # Shape function subplot
            ax_shape = axes[row, 0]
            if shape_func['datatype'] == 'numerical':
                if 'x' in shape_func and 'y' in shape_func and len(shape_func['x']) > 0 and len(shape_func['y']) > 0:
                    # Plot y values in scaled units (no denormalization)
                    ax_shape.plot(shape_func['x'], shape_func['y']*1000, linewidth=3, color='black')
                ax_shape.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
                feature_label = pretty_names.get(feature_name, feature_name)
                # Add units to x-axis labels based on feature type
                if feature_name == 'temp':
                    ax_shape.set_xlabel(f'{feature_label} (°C)', fontsize=14)
                elif feature_name == 'hum':
                    ax_shape.set_xlabel(f'{feature_label} (%)', fontsize=14)
                elif feature_name == 'windspeed':
                    ax_shape.set_xlabel(f'{feature_label} (km/h)', fontsize=14)
                else:
                    ax_shape.set_xlabel(f'{feature_label}', fontsize=14)
                ax_shape.set_ylabel('Bike Rentals/Day', fontsize=14)
                ax_shape.set_ylim(y_min*1000, y_max*1000)
                #ax_shape.set_title(f'{feature_label} - Shape Function', fontsize=16, fontweight='bold')
                ax_shape.legend()
                
                # Enhanced grid settings with custom spacing
                from matplotlib.ticker import MultipleLocator
                
                # Set custom y-axis grid spacing (every 500 for major, 250 for minor)
                y_major_locator = MultipleLocator(500)  # Major grid every 500 bikes
                y_minor_locator = MultipleLocator(250)  # Minor grid every 250 bikes
                ax_shape.yaxis.set_major_locator(y_major_locator)
                ax_shape.yaxis.set_minor_locator(y_minor_locator)
                
                # Set custom x-axis grid spacing based on feature type
                if feature_name == 'temp':
                    x_major_locator = MultipleLocator(5)   # Every 5°C
                    x_minor_locator = MultipleLocator(2.5) # Every 2.5°C
                elif feature_name == 'hum':
                    x_major_locator = MultipleLocator(10)  # Every 10%
                    x_minor_locator = MultipleLocator(5)   # Every 5%
                elif feature_name == 'windspeed':
                    x_major_locator = MultipleLocator(5)   # Every 5 km/h
                    x_minor_locator = MultipleLocator(2.5) # Every 2.5 km/h
                else:
                    # Default for other features
                    x_major_locator = MultipleLocator(5)
                    x_minor_locator = MultipleLocator(2.5)
                
                ax_shape.xaxis.set_major_locator(x_major_locator)
                ax_shape.xaxis.set_minor_locator(x_minor_locator)
                
                # Major grid lines
                ax_shape.grid(True, which='major', alpha=0.3, linewidth=0.8)
                # Minor grid lines (adds lines in between)
                ax_shape.grid(True, which='minor', alpha=0.15, linewidth=0.4)
                ax_shape.minorticks_on()  # Enable minor ticks
        
        # Ensure all subplots are properly configured
        for row in range(n_rows):
            for col in range(n_cols):
                if row < n_features:
                    # Make sure the subplot is visible and has proper formatting
                    axes[row, col].tick_params(axis='both', which='major', labelsize=12)
                    axes[row, col].spines['top'].set_visible(True)
                    axes[row, col].spines['right'].set_visible(True)
                    axes[row, col].spines['bottom'].set_visible(True)
                    axes[row, col].spines['left'].set_visible(True)
        
        plt.tight_layout(pad=2.0)  
        return fig
    except Exception as e:
        print(f"Error in create_numerical_shape_plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_plots(model, X_test, y_test_scaled, base_model=None):
    """Create shape functions plot only"""
    try:
        # If model is None, return None (no plot before retrain)
        if model is None:
            return None
        fig_shape = create_numerical_shape_plot(model)
        if fig_shape is None:
            print("Warning: create_numerical_shape_plot returned None")
            return None
        shape_buffer = io.BytesIO()
        fig_shape.savefig(shape_buffer, format='png', bbox_inches='tight', dpi=150)
        shape_buffer.seek(0)
        shape_plot = base64.b64encode(shape_buffer.getvalue()).decode()
        plt.close(fig_shape)
        return shape_plot
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_metrics(model, X_test, y_test_scaled):
    """Calculate model performance metrics"""
    y_pred = model.predict(X_test)
    
    y_pred_original = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    
    return {'r2': r2, 'mse': mse, 'mae': mae}

# Calculate base metrics
base_metrics = calculate_metrics(base_model, X_test, y_test_scaled)

app = Flask(__name__)
app.secret_key = os.environ.get('HIL_XAI_APP_SECRET_KEY') or secrets.token_hex(32)

def generate_subject_id():
    """Generate a unique subject ID"""
    return str(uuid.uuid4())[:8]

def get_subject_data():
    """Get or create subject data"""
    if 'subject_id' not in session:
        session['subject_id'] = generate_subject_id()
        session['start_time'] = datetime.now().isoformat()
        session['page_times'] = {}
        session['current_page_start'] = datetime.now().isoformat()
    
    return {
        'subject_id': session['subject_id'],
        'start_time': session['start_time'],
        'page_times': session.get('page_times', {}),
        'current_page_start': session.get('current_page_start', datetime.now().isoformat())
    }

def update_page_time(page_name):
    """Update time spent on current page and start timing new page"""
    if 'current_page_start' in session and 'page_times' in session:
        current_start = datetime.fromisoformat(session['current_page_start'])
        time_spent = (datetime.now() - current_start).total_seconds()
        
        if page_name in session['page_times']:
            session['page_times'][page_name] += time_spent
        else:
            session['page_times'][page_name] = time_spent
    
    session['current_page_start'] = datetime.now().isoformat()
    session.modified = True

@app.route("/")
def introduction():
    subject_data = get_subject_data()
    update_page_time('introduction')
    return render_template_string(introduction_html, request=request)

@app.route("/questionnaire")
def questionnaire():
    subject_data = get_subject_data()
    update_page_time('questionnaire')
    return render_template_string(questionnaire_html, request=request)

@app.route("/main")
def main():
    subject_data = get_subject_data()
    update_page_time('main')
    return render_template_string(
        html,
        predictors=predictors,
        base_metrics=base_metrics,
        request=request
    )

@app.route("/questionnaire-continuation")
def questionnaire_continuation():
    subject_data = get_subject_data()
    update_page_time('questionnaire_continuation')
    return render_template_string(questionnaire_continuation_html, request=request)

@app.route("/thank-you")
def thank_you():
    subject_data = get_subject_data()
    update_page_time('thank_you')
    return render_template_string(thank_you_html, request=request)

@app.route("/get-subject-info")
def get_subject_info():
    """Get current subject information"""
    subject_data = get_subject_data()
    return jsonify(subject_data)

@app.route("/submit-questionnaire", methods=["POST"])
def submit_questionnaire():
    """Handle questionnaire data submission"""
    try:
        data = request.get_json()
        subject_data = get_subject_data()
        
        # Server-side validation
        required_fields = {
            'field': 'Field',
            'education': 'Education Level', 
            'ml_familiarity': 'Machine Learning Familiarity',
            'interpretable_ml_familiarity': 'Interpretable ML Familiarity',
            'chart_comfort': 'Chart Reading Comfort',
            'bike_sharing_familiarity': 'Bike Sharing Familiarity'
        }
        
        missing_fields = []
        for field, label in required_fields.items():
            # For all fields (radio buttons, select elements)
            if field not in data or not data[field] or data[field] == '':
                missing_fields.append(label)
        

        
        if missing_fields:
            return jsonify({
                "status": "error", 
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Only save file if all validation passes
        # Update final page time
        update_page_time('questionnaire')
        
        # Add subject and timing information
        data['subject_id'] = subject_data['subject_id']
        data['timestamp'] = datetime.now().isoformat()
        data['questionnaire_type'] = 'pre_study'
        data['page_times'] = subject_data['page_times']
        data['total_time'] = (datetime.now() - datetime.fromisoformat(subject_data['start_time'])).total_seconds()
        
        # Save to subject-specific file
        import json
        import os
        
        # Create responses directory if it doesn't exist
        os.makedirs(RESPONSES_DIR, exist_ok=True)
        
        filename = os.path.join(RESPONSES_DIR, f"subject_{subject_data['subject_id']}.json")
        
        # Load existing data if file exists, otherwise create new
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                subject_file_data = json.load(f)
        else:
            subject_file_data = {
                'subject_id': subject_data['subject_id'],
                'start_time': subject_data['start_time'],
                'pre_study_survey': {},
                'post_study_survey': {},
                'page_times': {},
                'model_interactions': []
            }
        
        # Update with pre-study data
        subject_file_data['pre_study_survey'] = data
        subject_file_data['page_times'] = subject_data['page_times']
        
        with open(filename, 'w') as f:
            json.dump(subject_file_data, f, indent=2)
        
        print(f"Pre-study Survey data saved to: {filename}")
        return jsonify({"status": "success", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/submit-continuation", methods=["POST"])
def submit_continuation():
    """Handle post-study questionnaire data submission"""
    try:
        data = request.get_json()
        subject_data = get_subject_data()
        
        # Server-side validation
        required_fields = {
            'trusted_predictions': 'Trust in System Predictions',
            'confident_relying': 'Confidence in Relying on System',
            'consistent_handling': 'System Consistency',
            'use_future': 'Future Use Intent',
            'hesitant_rely': 'Hesitation to Rely on System',
            'recommend_domain': 'Recommendation to Domain',
            'accept_decision_tool': 'Acceptance as Decision Tool',
            'trust_predictive_models': 'Trust in Predictive Models',
            'open_to_prediction_support': 'Openness to Prediction Support'
        }
        
        missing_fields = []
        for field, label in required_fields.items():
            if field not in data or not data[field] or data[field] == '':
                missing_fields.append(label)
        
        if missing_fields:
            return jsonify({
                "status": "error", 
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Update final page time
        update_page_time('questionnaire_continuation')
        
        # Add subject and timing information
        data['subject_id'] = subject_data['subject_id']
        data['timestamp'] = datetime.now().isoformat()
        data['questionnaire_type'] = 'post_study'
        data['page_times'] = subject_data['page_times']
        data['total_time'] = (datetime.now() - datetime.fromisoformat(subject_data['start_time'])).total_seconds()
        
        # Save to subject-specific file
        import json
        import os
        
        # Create responses directory if it doesn't exist
        os.makedirs(RESPONSES_DIR, exist_ok=True)
        
        filename = os.path.join(RESPONSES_DIR, f"subject_{subject_data['subject_id']}.json")
        
        # Load existing data if file exists, otherwise create new
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                subject_file_data = json.load(f)
        else:
            subject_file_data = {
                'subject_id': subject_data['subject_id'],
                'start_time': subject_data['start_time'],
                'pre_study_survey': {},
                'post_study_survey': {},
                'page_times': {},
                'model_interactions': []
            }
        
        # Update with post-study data
        subject_file_data['post_study_survey'] = data
        subject_file_data['page_times'] = subject_data['page_times']
        
        # Save plus_feedback if present
        if 'plus_feedback' in data and data['plus_feedback'].strip():
            subject_file_data['plus_feedback'] = data['plus_feedback'].strip()
        
        with open(filename, 'w') as f:
            json.dump(subject_file_data, f, indent=2)
        
        print(f"Post-study questionnaire data saved to: {filename}")
        return jsonify({"status": "success", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/track-interaction", methods=["POST"])
def track_interaction():
    """Track model interaction events"""
    try:
        data = request.get_json()
        subject_data = get_subject_data()
        
        # Add subject and timing information
        data['subject_id'] = subject_data['subject_id']
        data['timestamp'] = datetime.now().isoformat()
        
        # Save to subject-specific file
        import json
        import os
        
        filename = os.path.join(RESPONSES_DIR, f"subject_{subject_data['subject_id']}.json")
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                subject_file_data = json.load(f)
        else:
            subject_file_data = {
                'subject_id': subject_data['subject_id'],
                'start_time': subject_data['start_time'],
                'pre_study_survey': {},
                'post_study_survey': {},
                'page_times': {},
                'model_interactions': []
            }
        
        # Add interaction to list
        if 'model_interactions' not in subject_file_data:
            subject_file_data['model_interactions'] = []
        subject_file_data['model_interactions'].append(data)
        
        with open(filename, 'w') as f:
            json.dump(subject_file_data, f, indent=2)
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Train route not needed for control group

@app.route("/base-model-shapes", methods=["GET"])
def base_model_shapes():
    """Get shape functions for the base model (no constraints)"""
    try:
        # Create shape functions plot for base model only
        shape_plot = create_plots(base_model, X_test, y_test_scaled)
        
        return jsonify({
            "shape_plot": shape_plot
        })
        
    except Exception as e:
        print(f"Error in base-model-shapes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def generate_sample_predictions(model):
    """Generate predictions for 5 different combinations of all numerical features"""
    print(f"generate_sample_predictions called with model: {type(model)}")
    print(f"Model features: {model.numerical_cols}")
    
    # Create sample data with different combinations of all numerical features
    sample_data = []
    
    # Sample combinations using the actual training data scale (not normalized)
    # These values match the scale used in the shape function plots
    combinations = [
        (10, 30, 3),    # Cool, low humidity, light wind
        (20, 50, 8),    # Mild, moderate humidity, moderate wind
        (25, 60, 12),   # Moderate, moderate humidity, moderate wind
        (30, 70, 15),   # Warm, high humidity, moderate wind
        (35, 85, 25),   # Hot, very high humidity, strong wind
    ]
    
    # Original values for display (same as combinations since we're using actual scale)
    original_values = combinations
    
    for i, (temp_orig, hum_orig, wind_orig) in enumerate(combinations):
        # Create a sample with actual scale values (same scale as training data)
        sample = pd.DataFrame({
            'temp': [temp_orig],        # Actual temperature (°C)
            'hum': [hum_orig],          # Actual humidity (%)
            'windspeed': [wind_orig]    # Actual wind speed (km/h)
        })
        
        print(f"Sample {i+1}: {sample}")
        
        # Make prediction
        try:
            prediction_scaled = model.predict(sample)[0]*1000
            print(f"Prediction {i+1}: {prediction_scaled}")
        except Exception as e:
            print(f"Error making prediction {i+1}: {e}")
            import traceback
            traceback.print_exc()
            prediction_scaled = 0  # fallback value
        
        #prediction_original = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        
        sample_data.append({
            'temperature': temp_orig,
            'humidity': hum_orig,
            'windspeed': wind_orig,
            'prediction': prediction_scaled
        })
    
    print(f"Final sample_data: {sample_data}")
    return sample_data



@app.route("/save-prediction-feedback", methods=["POST"])
def save_prediction_feedback():
    """Save user feedback on predictions"""
    try:
        data = request.get_json()
        subject_data = get_subject_data()
        
        # Add subject and timing information
        data['subject_id'] = subject_data['subject_id']
        data['timestamp'] = datetime.now().isoformat()
        data['feedback_type'] = 'prediction_feedback'
        
        # Save to subject-specific file
        import json
        import os
        
        filename = os.path.join(RESPONSES_DIR, f"subject_{subject_data['subject_id']}.json")
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                subject_file_data = json.load(f)
        else:
            subject_file_data = {
                'subject_id': subject_data['subject_id'],
                'start_time': subject_data['start_time'],
                'pre_study_questionnaire': {},
                'post_study_questionnaire': {},
                'page_times': {},
                'model_interactions': [],
                'prediction_feedback': []
            }
        
        # Add prediction feedback to list
        if 'prediction_feedback' not in subject_file_data:
            subject_file_data['prediction_feedback'] = []
        subject_file_data['prediction_feedback'].append(data)
        
        with open(filename, 'w') as f:
            json.dump(subject_file_data, f, indent=2)
        
        print(f"Prediction feedback saved to: {filename}")
        return jsonify({"status": "success", "filename": filename})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Network configuration
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8050"))

def get_app():
    return app

@app.route("/base-model-predictions", methods=["GET"])
def base_model_predictions():
    """Return sample predictions from the base model only"""
    try:
        print("base-model-predictions endpoint called")
        print(f"Base model features: {base_model.numerical_cols}")
        print(f"Base model trained: {hasattr(base_model, 'linear_model')}")
        predictions = generate_sample_predictions(base_model)
        print(f"Generated predictions: {predictions}")
        return jsonify({"predictions": predictions})
    except Exception as e:
        print(f"Error in base-model-predictions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["GET"])
def test():
    """Simple test endpoint"""
    return jsonify({"status": "ok", "message": "Control group server is running"})