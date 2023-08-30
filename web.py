from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open(r'C:\Users\USER\Desktop\DSA course\HR AV hacakthon\Web app\best_random_forest_model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index_main.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 
        'EducationField', 'EmployeeCount', 'Gender', 'JobLevel', 'JobRole', 
        'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 
        'PercentSalaryHike', 'StandardHours', 'StockOptionLevel', 
        'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 
        'YearsSinceLastPromotion', 'YearsWithCurrManager', 
        'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 
        'JobInvolvement', 'PerformanceRating'
    ]
    
    input_data = [request.form[feature] for feature in features]
    input_data = np.array(input_data, dtype=float).reshape(1, -1)
    
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        result = "No"
    else:
        result = "Yes"
    
    return render_template('result.html', prediction_text=f"Predicted Attrition: {result}")

if __name__ == '__main__':
    app.run(debug=True)
