import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
from dsapp.models import ModelInfo3
from sklearn.metrics import accuracy_score, recall_score, classification_report

warnings.filterwarnings('ignore')

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        
        df = pd.read_csv('activity_logs_detailed.csv')

        df['type_name'] = df['type_name'].str.lower().str.replace(" ", "_")
        df['day'] = df['day'].str.lower().str.replace(" ", "_")

        df_encoded = pd.get_dummies(df, columns=['type_name', 'day', 'duration_category'])

        feature_columns = [
            'hour', 'minute', 'day_of_week', 'minutes_since_midnight',
            'is_weekend', 'is_morning', 'is_afternoon', 'is_evening',
            'is_peak', 'is_peak_or_weekend', 'is_early_morning',
            'daily_activity_count'
        ] + [col for col in df_encoded.columns if col.startswith('type_name_') or 
             col.startswith('day_') or col.startswith('duration_category_')]

       
        df['engaged'] = (df['duration'] > 15).astype(int)

        x = df_encoded[feature_columns]
        y = df['engaged']
    
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        model.fit(x_train, y_train)
      
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
      
        print(classification_report(y_test, y_pred))

        model_filename = 'final_duration_predictor_rf.pkl'
        joblib.dump(model, model_filename)
        self.stdout.write(self.style.SUCCESS(f'Model saved as {model_filename}'))

        model_info = ModelInfo3.objects.create(
            model_name='Random Forest',
            model_file=model_filename,
            training_data='activity_logs_detailed.csv',
            training_date=pd.Timestamp.now(tz='Asia/Jakarta'),
            model_summary=f"Model trained with Accuracy: {accuracy:.3f} & Recall: {recall:.3f}",
            creator='Rich',
            usecase='Optimal Schedule Recommendation for Student Activities Based on Predicted Student Participation'
        )
        self.stdout.write(self.style.SUCCESS(f'Model info saved to database: ID {model_info.id}'))
