import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from dsapp.models import Student, Enrollment, CourseActivity, StudentActivityLog

class Command(BaseCommand):
    help = 'ETL: Extract, Transform, and Load student activity data for ML'
    
    def handle(self, *args, **kwargs):
        data = []
        
        course_activity_map = {
            ca.activity_id: ca.type_id
            for ca in CourseActivity.objects.all()
        }
        
        #Age 
        for student in Student.objects.all():
            today = datetime.now().date()
            age = (today - student.dob).days // 365
            
            # Get all activity logs for this student
            logs = StudentActivityLog.objects.filter(stu_id=student)
            
            type_summary = {}
            for log in logs:
                type_id = course_activity_map.get(log.activity_id_id)
                if type_id is None:
                    continue # Skip if type_id is not found
                
                duration = (log.activity_end - log.activity_start).total_seconds() / 60
                key = f"time_type_{type_id}"
                type_summary[key] = type_summary.get(key, 0) + duration
                
            # Calculate average grade for this student
            enrollment = Enrollment.objects.filter(stu_id=student)
            grades = [e.grade for e in enrollment if e.grade is not None]
            avg_grade = sum(grades) / len(grades) if grades else None
            
            record = {
                "stu_id": student.stu_id,
                "gender": student.gender,
                "age": age,
                "avg_grade": avg_grade,
            }
            record.update(type_summary) # add time_type_* columns
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        df.to_csv('student_activity_dataset.csv', index=False)
        self.stdout.write(self.style.SUCCESS('Student activity dataset saved to CSV!'))