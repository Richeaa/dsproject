import pandas as pd
import json
import joblib
import os
from django.shortcuts import render
from django.core.paginator import Paginator
from .models import PredictionResult
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Create your views here.
def home (request):
    return render(request, 'home.html')


def about(request):
    return render(request, 'about.html')

def dashboard (request):
    return render(request, 'dashboard.html')

# def view_data (request):
#     students = Student.objects.all()
#     today = datetime.now().date()
    
#     # Age
#     for student in students:
#         dob = student.dob
#         student.age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    
#     #paginator
#     paginator = Paginator(students, 10)
#     page_number = request.GET.get('page')
#     students = paginator.get_page(page_number)
    
#     return render(request, 'view_data.html', {'students': students})

def view_learning_path (request):
    return render(request, 'view_learning_path.html')

@csrf_exempt
def get_learning_path_data(request):
    if request.method == "POST":
        try:
            # Parse incoming JSON data
            data = json.loads(request.body)
            
            selected_model = data.get("model", "xgboost")  # default to xgboost
            if selected_model == "random_forest":
                model_path = os.path.join(settings.BASE_DIR, 'rfr_learning_path_model.pkl')
            else:
                model_path = os.path.join(settings.BASE_DIR, 'xgb_learning_path_model.pkl')
                
            model = joblib.load(model_path)
                    
            df = pd.read_csv(os.path.join(settings.BASE_DIR, 'stu_activ_recom.csv')).fillna(0)

            # Rename columns for consistency
            df.rename(columns={
                'time_type_ActivityType object (1)': 'time_Quiz',
                'time_type_ActivityType object (2)': 'time_IndividualAssignment',
                'time_type_ActivityType object (3)': 'time_GroupAssignment',
                'time_type_ActivityType object (4)': 'time_Forum'
            }, inplace=True)

            # Encode gender
            le = LabelEncoder()
            df["gender_encoded"] = le.fit_transform(df["gender"])
            
            # Map gender values
            gender_mapping = {'Male': 0, 'Female': 1}

            # Define activity columns
            activity_cols = ['time_Quiz', 'time_IndividualAssignment', 'time_GroupAssignment', 'time_Forum']
            features = ["age", "gender_encoded"] + activity_cols

            # Check if this is a new prediction or existing student lookup
            if data.get("new_prediction", False):
                # Handle new prediction
                # Create features for the prediction
                new_features = {
                    "age": data.get("age", 20),
                    "gender_encoded": gender_mapping.get(data.get("gender", "Male"), 0),
                    "time_Quiz": data.get("time_Quiz", 0),
                    "time_IndividualAssignment": data.get("time_IndividualAssignment", 0),
                    "time_GroupAssignment": data.get("time_GroupAssignment", 0),
                    "time_Forum": data.get("time_Forum", 0)
                }
                
                student_features = pd.Series(new_features)
                student_df = pd.DataFrame([student_features.values], columns=features)
                overall_prediction = float(model.predict(student_df)[0])
                
                # Per-activity predictions
                activity_predictions = {}
                for col in activity_cols:
                    test_row = student_features.copy()
                    test_row[activity_cols] = 0
                    test_row[col] = student_features[col]
                    test_df = pd.DataFrame([test_row.values], columns=features)
                    pred = float(model.predict(test_df)[0])
                    activity_predictions[col.replace("time_", "")] = pred

                # Activity time usage
                activity_times = {col.replace("time_", ""): float(new_features[col]) for col in activity_cols}

                # Get student's completed activities
                student_activities = set([
                    col.replace("time_", "") for col in activity_cols if new_features[col] > 0
                ])
                
                # Generate association rules
                apriori_rules = generate_apriori_rules(df, activity_cols, student_activities)

                # Response for new prediction
                response_data = {
                    "student_id": None,
                    "actual_grade": None,
                    "predicted_grade": overall_prediction,
                    "activity_predictions": activity_predictions,
                    "activity_times": activity_times,
                    "student_activities": list(student_activities),
                    "association_rules": apriori_rules
                }
                
            else:
                # Handle existing student lookup
                student_id = int(data.get("student_id", 1))  # Default to student 1 if not provided
                
                # Find student
                student = df[df["stu_id"] == student_id]
                if student.empty:
                    return JsonResponse({"error": "Student not found"}, status=404)

                actual_grade = float(student.iloc[0]["avg_grade"])
                student_features = student.iloc[0][features]
                student_df = pd.DataFrame([student_features.values], columns=features)
                overall_prediction = float(model.predict(student_df)[0])

                # Per-activity predictions
                activity_predictions = {}
                for col in activity_cols:
                    test_row = student_features.copy()
                    test_row[activity_cols] = 0
                    test_row[col] = student_features[col]
                    test_df = pd.DataFrame([test_row.values], columns=features)
                    pred = float(model.predict(test_df)[0])
                    activity_predictions[col.replace("time_", "")] = pred

                # Activity time usage
                activity_times = {col.replace("time_", ""): float(student.iloc[0][col]) for col in activity_cols if not pd.isna(student.iloc[0][col])}

                # Get student's completed activities
                student_activities = set([
                    col.replace("time_", "") for col in activity_cols if student.iloc[0][col] > 0
                ])
                
                # Generate association rules
                apriori_rules = generate_apriori_rules(df, activity_cols, student_activities)

                # Response for existing student
                response_data = {
                    "student_id": student_id,
                    "actual_grade": actual_grade,
                    "predicted_grade": overall_prediction,
                    "activity_predictions": activity_predictions, 
                    "activity_times": activity_times,
                    "student_activities": list(student_activities),
                    "association_rules": apriori_rules
                }
                
            return JsonResponse(response_data)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


def generate_apriori_rules(df, activity_cols, student_activities):
    try:
        df_binary = df[['stu_id'] + activity_cols].copy()
        for col in activity_cols:
            df_binary[col] = df_binary[col].apply(lambda x: 1 if x > 0 else 0)

        transactions = []
        for _, row in df_binary.iterrows():
            activities = [col.replace('time_', '') for col in activity_cols if row[col] == 1]
            if activities:
                transactions.append(activities)

        if not transactions:
            return []

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        te_df = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = apriori(te_df, min_support=0.1, use_colnames=True)
        if frequent_itemsets.empty:
            return []

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        rules = rules[(rules['support'] >= 0.1) & (rules['lift'] > 1.04)]

        # Personalize: filter rules where student has done all the antecedents
        personalized_rules = []
        for _, row in rules.iterrows():
            antecedents = set(row['antecedents'])
            if antecedents.issubset(student_activities):  # This is the key filter
                # Convert to list and shuffle the order of activities
                antecedent_list = list(antecedents)
                consequent_list = list(row['consequents'])
                random.shuffle(antecedent_list)
                random.shuffle(consequent_list)
                
                personalized_rules.append({
                    "antecedent": ', '.join(antecedent_list),
                    "consequent": ', '.join(consequent_list),
                    "confidence": float(round(row['confidence'], 2)),
                    "support": float(round(row['support'], 2)),
                    "lift": float(round(row['lift'], 2)),
                })

        personalized_rules = sorted(
            personalized_rules,
            key=lambda x: (-x["confidence"], -x["lift"], -x["support"])
        )[:5]

        return personalized_rules
    except Exception as e:
        print(f"Error generating association rules: {str(e)}")
        return []
    
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