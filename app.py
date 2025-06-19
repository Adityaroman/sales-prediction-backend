from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

try:
    model = joblib.load('sales_model.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_marital = joblib.load('le_marital.pkl')
    le_state = joblib.load('le_state.pkl')
    le_category = joblib.load('le_category.pkl')
    le_age_group = joblib.load('le_age_group.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or encoders: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([{
            'Age': int(data['Age']),
            'Gender': le_gender.transform([data['Gender']])[0],
            'Marital_Status': le_marital.transform([data['Marital_Status']])[0],
            'State': le_state.transform([data['State']])[0],
            'Product_Category': le_category.transform([data['Product_Category']])[0],
            'Age_Group': le_age_group.transform([data['Age_Group']])[0],
            'Orders': int(data['Orders'])
        }])
        prediction = model.predict(df)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/eda', methods=['GET'])
def eda():
    try:
        df = pd.read_csv('data/sales_data.csv')
        df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 70], labels=['18-25', '26-35', '36-45', '46-55', '56-70'])
        sales_gen = df.groupby('Gender')['Amount'].sum().to_dict()
        sales_state = df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(10).to_dict()
        sales_category = df.groupby('Product_Category')['Amount'].sum().sort_values(ascending=False).to_dict()
        sales_age_group = df.groupby('Age_Group')['Amount'].sum().to_dict()
        return jsonify({
            'sales_by_gender': sales_gen,
            'sales_by_state': sales_state,
            'sales_by_category': sales_category,
            'sales_by_age_group': sales_age_group
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)