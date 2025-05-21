from django.shortcuts import render
import json
import joblib
import os
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
from .models import PredictionResult
from django.db.models import Q

def home(request):
    return render(request, 'home.html')


def about(request):
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

        average_grade =  round((course1 + course2 + course3 + course4 + course5) / 5, 2)

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
    
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
def predict_cluster(request):
    if request.method == 'POST':
        student_id = request.POST.get('student_id')
        total_study_time = float(request.POST.get('total_study_time'))
        avg_study_time = float(request.POST.get('avg_study_time'))
        activity_count = float(request.POST.get('activity_count'))
        active_days = float(request.POST.get('active_days'))

        data = np.array([[total_study_time, avg_study_time, activity_count, active_days]])
        scaled_data = scaler.transform(data)
        cluster = kmeans_model.predict(scaled_data)[0]
        pca_result = pca.transform(scaled_data)[0]

        if cluster == 1:
            analysis = "This student shows minimal engagement with very little LMS usage."
            recommendation = "Encourage the student through personalized support and reminders to access learning materials regularly."
        elif cluster == 0 and total_study_time > 10:
            analysis = "This student is highly engaged with consistent LMS usage."
            recommendation = "Provide advanced tasks or self-paced learning modules to maintain motivation."

        else:
            analysis = "This student shows moderate engagement with steady learning patterns."
            recommendation = "Maintain current learning pace and give weekly feedback."


        # Simpan plot visualisasi
        generate_cluster_plot(pca, kmeans_model, scaler, pca_result)

        # Simpan hasil ke DB
        PredictionResult.objects.create(
            student_id=student_id,
            total_study_time=total_study_time,
            avg_study_time=avg_study_time,
            activity_count=activity_count,
            active_days=active_days,
            cluster=cluster,
            pca_x=pca_result[0],
            pca_y=pca_result[1],
        )

        context = {
            'cluster': cluster,
            'pca_x': round(pca_result[0], 2),
            'pca_y': round(pca_result[1], 2),
            'plot_path': 'static/cluster_plot.png',
            'student_id': student_id,
            'total_study_time': total_study_time,
            'avg_study_time': avg_study_time,
            'activity_count': activity_count,
            'active_days': active_days,
            'analysis': analysis,
            'recommendation': recommendation,

        }

        return render(request, 'result.html', context)

    return render(request, 'predict_form.html')


def generate_cluster_plot(pca, kmeans_model, scaler, user_point_pca):
    # Load ulang dataset lama
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'grade_dataset.csv'))
    features = ['total_study_time', 'avg_study_time', 'activity_count', 'active_days']
    X_scaled = scaler.transform(df[features])
    X_pca = pca.transform(X_scaled)
    labels = kmeans_model.predict(X_scaled)

    # Buat DataFrame hasil PCA
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=50)
    
    # Titik user
    plt.scatter(user_point_pca[0], user_point_pca[1], c='black', s=100, marker='X', label='Input Mahasiswa')
    plt.legend()
    plt.title("Visualisasi Clustering Mahasiswa (PCA)")
    
    # Simpan ke static
    plot_path = os.path.join(settings.BASE_DIR, 'dsapp/static/', 'cluster_plot.png')
    plt.savefig(plot_path)
    plt.close()

def get_cluster_characteristics():
    from .models import PredictionResult
    import pandas as pd

    queryset = PredictionResult.objects.all().values()
    df = pd.DataFrame(queryset)

    if df.empty:
        return []

    grouped = df.groupby('cluster').agg({
        'total_study_time': 'mean',
        'avg_study_time': 'mean',
        'activity_count': 'mean',
        'active_days': 'mean',
        'cluster': 'count'
    }).rename(columns={'cluster': 'count'})

    characteristics = []
    for idx, row in grouped.iterrows():
        characteristics.append({
            'cluster': idx,
            'count': int(row['count']),
            'total_study_time': round(row['total_study_time'], 2),
            'avg_study_time': round(row['avg_study_time'], 2),
            'activity_count': round(row['activity_count'], 2),
            'active_days': round(row['active_days'], 2),
        })

    return characteristics

def search_cluster(request):
    query = request.GET.get('student_id')  # ⬅️ make sure form uses this name
    result = None
    analysis = None
    recommendation = None
    if query:
        result = PredictionResult.objects.filter(student_id=query).last()

        if result:
            # regenerate PCA plot for the found student
            data = np.array([[result.total_study_time, result.avg_study_time, result.activity_count, result.active_days]])
            scaled = scaler.transform(data)
            pca_point = pca.transform(scaled)[0]
            generate_cluster_plot(pca, kmeans_model, scaler, pca_point)

            # ANALYSIS & RECOMMENDATION based on cluster
            if result.cluster == 1:
                analysis = "This student shows minimal engagement with very little LMS usage."
                recommendation = "Encourage the student through personalized support and reminders to access learning materials regularly."
            elif result.cluster == 0 and result.total_study_time > 10:
                analysis = "This student is highly engaged with consistent LMS usage."
                recommendation = "Provide advanced tasks or self-paced learning modules to maintain motivation."
            else:
                analysis = "This student shows moderate engagement with steady learning patterns."
                recommendation = "Maintain current learning pace and give weekly feedback."

    return render(request, 'search_result.html', {
        'result': result,
        'plot_path': 'static/cluster_plot.png',
        'analysis': analysis,
        'recommendation': recommendation
})
