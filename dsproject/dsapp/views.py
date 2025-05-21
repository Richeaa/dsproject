from django.shortcuts import render
import json
import joblib
import os
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def home (request):
    return render(request, 'home.html')

def about (request):
    return render(request, 'about.html')

def dashboard (request):
    return render(request, 'dashboard.html')

def underperform(request):
    df = pd.read_csv('underperform_students.csv')
    df_html = df.to_html(index=False)
    return render(request, 'underperform.html', {'df_html': df_html})



MODEL_PATHS = {
    'lr': os.path.join(settings.BASE_DIR, 'final_underperform_students_lr.pkl'),
    'rf': os.path.join(settings.BASE_DIR, 'final_underperform_students_rf.pkl'),
}

@csrf_exempt
def predict_status(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        model_name = data.get('model_name')
        if model_name not in MODEL_PATHS:
            return JsonResponse({'error': 'Invalid model selected'}, status=400)

        model_path = MODEL_PATHS[model_name]
        model = joblib.load(model_path)

        def safe_float(value):
            try:
                if value is None or value == '':
                    return 0
                return float(value)
            except (ValueError, TypeError):
                return 0

        course1 = safe_float(data['course1'])
        course2 = safe_float(data['course2'])
        course3 = safe_float(data['course3'])
        course4 = safe_float(data['course4'])
        course5 = safe_float(data['course5'])

        average_grade = (course1 + course2 + course3 + course4 + course5) / 5

        features = np.array([
            course1,
            course2,
            course3,
            course4,
            course5,
            average_grade
        ]).reshape(1, -1)

        dffeatures = pd.DataFrame(features, columns =[
            'course1',
            'course2',
            'course3',
            'course4',
            'course5',
            'average_grade'
        ])

        prediction = model.predict(dffeatures)[0]
        probability = model.predict_proba(dffeatures)[0].tolist()

        return JsonResponse({
            "prediction": int(prediction),
            "probability": probability,
            "avg_grade": average_grade,
            "advice": "Consider mentoring for better performance" if prediction == 0 else "Keep up the good work!"
        })
