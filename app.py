from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Load CNN model and scaler
cnn_model = load_model('models/cnn_model.h5')
scaler = joblib.load('models/scaler.pkl')

# SQLite database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", ('admin', 'password123'))
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Load training data columns for alignment
data = pd.read_csv('data/creditcard.csv')
X = data[['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'category', 'gender']]
X = pd.get_dummies(X, columns=['category', 'gender'], drop_first=True)
feature_columns = X.columns

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()
        if result and result[0] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials!', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'error')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/index')
def index():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    
    data = {
        'amt': float(request.form['amt']),
        'lat': float(request.form['lat']),
        'long': float(request.form['long']),
        'city_pop': float(request.form['city_pop']),
        'unix_time': float(request.form['unix_time']),
        'merch_lat': float(request.form['merch_lat']),
        'merch_long': float(request.form['merch_long']),
        'category': request.form['category'],
        'gender': request.form['gender']
    }
    
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df, columns=['category', 'gender'], drop_first=True)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    
    data_scaled = scaler.transform(input_df)
    
    # CNN Prediction
    data_cnn = data_scaled.reshape((1, data_scaled.shape[1], 1))
    cnn_prob = cnn_model.predict(data_cnn, verbose=0)[0][0]
    print(f"CNN raw probability: {cnn_prob}")
    cnn_pred = 1 if cnn_prob > 0.1 else 0
    cnn_result = 'Fraudulent' if cnn_pred == 1 else 'Legitimate'
    
    # Graph 1: Model Prediction with Legitimate and Fraudulent bars
    plt.figure(figsize=(6, 4))
    labels = ['Legitimate', 'Fraudulent']
    values = [1 - cnn_prob, cnn_prob]  # Split probability into two parts
    colors = ['blue', 'red']  # Blue for Legitimate, Red for Fraudulent
    bars = plt.bar(labels, values, color=colors, width=0.4)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom')
    
    plt.ylim(0, 1)
    plt.ylabel('Probability')
    plt.title('Model Prediction Probabilities')
    
    img1 = io.BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight')
    img1.seek(0)
    graph1_url = base64.b64encode(img1.getvalue()).decode('utf8')
    plt.close()
    
    # Graph 2: Feature Values
    key_features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    feature_values = [data_scaled[0][input_df.columns.get_loc(f)] for f in key_features]

    # Normalize feature values for better visualization (scale to [-1, 1] range)
    max_abs_value = max(abs(min(feature_values)), abs(max(feature_values)))
    if max_abs_value != 0:  # Avoid division by zero
        normalized_values = [v / max_abs_value for v in feature_values]
    else:
        normalized_values = feature_values

    plt.figure(figsize=(10, 6))
    bars = plt.bar(key_features, normalized_values, color='green', width=0.6)

    # Add value labels on top of bars (showing original values)
    for bar, orig_value in zip(bars, feature_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{orig_value:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top')

    # Improve formatting
    plt.ylabel('Normalized Feature Value', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    plt.title('Normalized Input Features (Original Values Shown)', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Set y-axis limits for normalized values
    plt.ylim(-1.2, 1.2)

    plt.tight_layout()

    img2 = io.BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight', dpi=100)
    img2.seek(0)
    graph2_url = base64.b64encode(img2.getvalue()).decode('utf8')
    plt.close()
    
    result = {
        'cnn_result': cnn_result,
        'graph1_url': graph1_url,
        'graph2_url': graph2_url,
        'both_fraudulent': False
    }
    return render_template('results.html', result=result)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
