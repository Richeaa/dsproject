import pandas as pd
import json
import joblib
import os
from django.shortcuts import render
from django.core.paginator import Paginator
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from django.db.models import Sum, F, ExpressionWrapper, DurationField
from sklearn.cluster import KMeans
from dsapp.models import Student, StudentActivityLog
from django.http import HttpResponse
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
                    
            df = pd.read_csv(os.path.join(settings.BASE_DIR, 'student_activity_dataset.csv')).fillna(0)

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
    
def optimal(request):
    return render(request, 'optimal.html')



DAY_TO_INDEX = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6
}

@csrf_exempt
def predict_schedule(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        activity_type = data.get('type_name', '').strip().lower().replace(" ", "_")
        day = data.get('day', '').strip().lower().replace(" ", "_")
        

        model_path = os.path.join(settings.BASE_DIR, 'final_duration_predictor_rf.pkl')
        model = joblib.load(model_path)

        if not activity_type or not day:
            return JsonResponse({'error': 'Missing required fields.'}, status=400)

        day_index = DAY_TO_INDEX[day.lower()]
        is_weekend = 1 if day.lower() in ['saturday', 'sunday'] else 0

        rows = []
        for hour in range(24):
            row = {
                'hour': hour,
                'minute': 0,
                'day_of_week': day_index,
                'minutes_since_midnight': hour * 60,
                'is_weekend': is_weekend,
                'is_morning': 1 if 6 <= hour <= 11 else 0,
                'is_afternoon': 1 if 12 <= hour <= 17 else 0,
                'is_evening': 1 if 18 <= hour <= 22 else 0,
                'is_peak': 1 if hour in [19, 20, 21] else 0,
                'is_peak_or_weekend': 1 if hour in [19, 20, 21] or is_weekend else 0,
                'is_early_morning': 1 if 0 <= hour <= 5 else 0,
                'daily_activity_count': 0,  
                f'type_name_{activity_type}': 1,
                f'day_{day}': 1,
            }
            rows.append(row)

        df = pd.DataFrame(rows).fillna(0)
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0  

        proba = model.predict_proba(df[model.feature_names_in_])[:, 1] 
        best_hour = int(np.argmax(proba))
        best_score = float(proba[best_hour])

        top3_indices = np.argsort(proba)[-5:][::-1]  
        top3_scores = proba[top3_indices]

        top3_score = [
            {'hour': int(hour), 'score': round(float(score), 3)}
            for hour, score in zip(top3_indices, top3_scores)
        ]

        return JsonResponse({
            'predicted_hour': best_hour,
            'engagement_score': round(best_score, 3),
            'all_hour_scores': [round(float(s), 3) for s in proba],
            'top3_score': top3_score
        })

    return JsonResponse({'error': 'Invalid request method.'}, status=405)
    
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
def generate_cluster_visualizations(input_study_time, input_avg_time):
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'grade_dataset.csv'))
    features = ['total_study_time', 'avg_study_time', 'activity_count', 'active_days']
    X = df[features]
    labels = kmeans_model.predict(scaler.transform(X))
    df['Cluster'] = labels

    # Donut Chart for Cluster Frequency
    cluster_counts = df['Cluster'].value_counts().sort_index()
    colors = sns.color_palette('Set2', len(cluster_counts))
    plt.figure(figsize=(6, 6))
    plt.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index], colors=colors,
            autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.4))
    plt.title('Cluster Distribution')
    plot_path = os.path.join(settings.BASE_DIR, 'dsapp/static/', 'cluster_donut.png')
    plt.savefig(plot_path)
    plt.close()

    # Scatter plot: total study time vs avg study time
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='total_study_time', y='avg_study_time', hue='Cluster', palette='Set2')
    plt.scatter(input_study_time, input_avg_time, color='black', s=100, marker='X', label='Input Student')
    plt.title('Student Clustering Based on Study Behavior')
    plt.xlabel('Total Study Time')
    plt.ylabel('Average Study Time')
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(settings.BASE_DIR, 'dsapp/static/', 'study_time_scatter.png')
    plt.savefig(scatter_path)
    plt.close()

    # Bar Charts for Activity Count and Active Days
    summary = df.groupby('Cluster')[['activity_count', 'active_days']].mean().reset_index()

    for feature in ['activity_count', 'active_days']:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=summary, x='Cluster', y=feature, palette='Set2')
        plt.title(f'Average {feature.replace("_", " ").title()} per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(feature.replace("_", " ").title())
        plt.tight_layout()
        path = os.path.join(settings.BASE_DIR, 'dsapp/static/', f'plot_{feature}.png')
        plt.savefig(path)
        plt.close()

def predict_cluster(request):
    if request.method == 'POST':
        total_study_time = float(request.POST.get('total_study_time'))
        avg_study_time = float(request.POST.get('avg_study_time'))
        activity_count = float(request.POST.get('activity_count'))
        active_days = float(request.POST.get('active_days'))

        data = np.array([[total_study_time, avg_study_time, activity_count, active_days]])

        # Latih ulang model dengan k yang dipilih
        df = pd.read_csv(os.path.join(settings.BASE_DIR, 'grade_dataset.csv'))
        features = ['total_study_time', 'avg_study_time', 'activity_count', 'active_days']
        X_scaled = scaler.transform(df[features])
        kmeans_model = KMeans(n_clusters=2, random_state=42)
        kmeans_model.fit(X_scaled)

        # Prediksi untuk input user
        scaled_data = scaler.transform(data)
        cluster = kmeans_model.predict(scaled_data)[0]

        # Generate ulang visualisasi
        generate_cluster_visualizations(total_study_time, avg_study_time)

        # Analisis & Rekomendasi
        if cluster == 1:
            analysis = "This student shows minimal engagement with very little LMS usage."
            recommendation = "Encourage the student through personalized support and reminders to access learning materials regularly."
        elif cluster == 0 and total_study_time > 10:
            analysis = "This student is highly engaged with consistent LMS usage."
            recommendation = "Provide advanced tasks or self-paced learning modules to maintain motivation."
        else:
            analysis = "This student shows moderate engagement with steady learning patterns."
            recommendation = "Maintain current learning pace and give weekly feedback."


        return render(request, 'result.html', {
            'cluster': cluster,
            'total_study_time': total_study_time,
            'avg_study_time': avg_study_time,
            'activity_count': activity_count,
            'active_days': active_days,
            'analysis': analysis,
            'recommendation': recommendation,
        })

    return render(request, 'predict_form.html')


def search_cluster(request):
    query = request.GET.get('student_id')
    result = None
    cluster = None
    features = None
    cluster_members = []

    if query:
        try:
            student = Student.objects.get(stu_id=query)
            logs = StudentActivityLog.objects.filter(stu_id=student)

            logs = logs.annotate(duration=ExpressionWrapper(F('activity_end') - F('activity_start'), output_field=DurationField()))
            total_study_time = logs.aggregate(total=Sum('duration'))['total']
            total_hours = round(total_study_time.total_seconds() / 3600, 2) if total_study_time else 0
            avg_hours = round(total_hours / logs.count(), 2) if logs.count() > 0 else 0
            activity_count = logs.count()
            active_days = logs.dates('activity_start', 'day').distinct().count()

            df = pd.read_csv(os.path.join(settings.BASE_DIR, 'grade_dataset.csv'))
            features_list = ['total_study_time', 'avg_study_time', 'activity_count', 'active_days']
            X = df[features_list]
            scaled_all = scaler.transform(X)

            kmeans_model = KMeans(n_clusters=2, random_state=42)
            kmeans_model.fit(scaled_all)

            data = np.array([[total_hours, avg_hours, activity_count, active_days]])
            scaled = scaler.transform(data)
            cluster = kmeans_model.predict(scaled)[0]

            generate_cluster_visualizations(total_hours, avg_hours)




            features = {
                'student_id': student.stu_id,
                'name': student.name,
                'total_study_time': total_hours,
                'avg_study_time': avg_hours,
                'activity_count': activity_count,
                'active_days': active_days,
                'cluster': cluster,
            }

            # Cari student lain dalam cluster yang sama
            all_students = Student.objects.exclude(stu_id=student.stu_id)
            for s in all_students:
                logs = StudentActivityLog.objects.filter(stu_id=s)
                logs = logs.annotate(duration=ExpressionWrapper(F('activity_end') - F('activity_start'), output_field=DurationField()))
                total = logs.aggregate(total=Sum('duration'))['total']
                total_h = round(total.total_seconds() / 3600, 2) if total else 0
                avg_h = round(total_h / logs.count(), 2) if logs.count() > 0 else 0
                count = logs.count()
                days = logs.dates('activity_start', 'day').distinct().count()

                d = np.array([[total_h, avg_h, count, days]])
                pred = kmeans_model.predict(scaler.transform(d))[0]

                if pred == cluster:
                    cluster_members.append({
                        'name': s.name,
                        'student_id': s.stu_id,
                        'total_study_time': total_h,
                        'avg_study_time': avg_h,
                        'activity_count': count,
                        'active_days': days
                    })

        except Student.DoesNotExist:
            student = None

    return render(request, 'search_result.html', {
        'result': features,
        'cluster': cluster,
        'cluster_members': cluster_members,
        'plot_path': 'static/cluster_plot.png',
    })
