import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from dsapp.models import ModelInfo3

class Command(BaseCommand):
    help = "Train a model to classify underperform students"

    def handle(self, *args, **kwargs):
        df = pd.read_csv('underperform_students.csv')
        
        df['average_label'] = (df['average_grade'] >=75).astype(int)

        features = ['course1', 'course2', 'course3', 'course4', 'course5', 'average_grade']
        x = df[features]
        y = df['average_label']

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=42)

        model = SVC(kernel ='rbf')
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        report = classification_report(y_test, predictions)
        self.stdout.write(self.style.SUCCESS("Classification report: \n" + report))

        model_filename = 'final_underperform_students_svm.pkl'
        joblib.dump(model, model_filename)
        self.stdout.write(self.style.SUCCESS(f'Model saved as {model_filename}'))


        model_info = ModelInfo3.objects.create(
            model_name='Support Vector Machines',
            model_file=model_filename,
            training_data = 'underperform_student.csv',
            training_date = pd.Timestamp.now(tz='Asia/Jakarta'),
            model_summary=report,
            creator = 'Rich',
            usecase = 'Allows lecturers to contact students to offer guidance, such as mentoring or study plans'
        )

        self.stdout.write(self.style.SUCCESS(f'Model info saved to database: ID {model_info}'))