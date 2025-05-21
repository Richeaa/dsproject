from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from .models import ModelInfo3
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def retrain_model_view(request, model_id):
    model_info = get_object_or_404(ModelInfo3, id=model_id)

    try:
        df = pd.read_csv('underperform_students.csv')

        df['average_label'] = (df['average_grade'] >= 75).astype(int)

        features = ['course1', 'course2', 'course3', 'course4', 'course5', 'average_grade']
        x = df[features]
        y = df['average_label']

        if model_info.model_name == 'Logistic Regression':
            model = LogisticRegression()
        elif model_info.model_name == 'Random Forest':
            model = RandomForestClassifier()
        else:
            messages.error(request, f"Unknown model type: {model_info.model_name}")
            return redirect('/admin/dsapp/modelinfo3/')

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=42)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred)
        
        model_path = f'{model_info.model_name}_retrained.pkl'
        joblib.dump(model, model_path)


        model_info.training_data = model_path,
        model_info.training_date = pd.Timestamp.now(tz='Asia/Jakarta')
        model_info.model_summary = report
        model_info.creator = 'Rich'
        model_info.usecase = 'Allows lecturers to contact students to offer guidance, such as mentoring or study plans'
        model_info.save()

        messages.success(request, f"Model '{model_info.model_name}' retrained successfully!")

    except Exception as e:
        messages.error(request, f"Error during retraining: {str(e)}")

    return redirect('/admin/dsapp/modelinfo3/')