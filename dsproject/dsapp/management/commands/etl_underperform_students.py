from django.core.management.base import BaseCommand
import pandas as pd
from django.db.models import Sum
from dsapp.models import Student, Enrollment

class Command(BaseCommand):
    help = "ETL: Extract student and course with grades and save as CSV"

    def handle(self, *args, **kwargs):
        student_data = []

        for student in Student.objects.all():
            enrollments = Enrollment.objects.filter(stu=student)

            total_grade = enrollments.aggregate(total=Sum('grade'))['total'] or 0
            average_grade = total_grade / 5 

            passes = 1 if average_grade >= 75 else 0

            course_grades = [0, 0, 0, 0, 0]

            for enrollment in enrollments:
                if 1 <= enrollment.course_id <= 5:
                    course_grades[enrollment.course_id - 1] = enrollment.grade
            
            student_record = {
                'student_name': student.name,
                'course1': course_grades[0],
                'course2': course_grades[1],
                'course3': course_grades[2],
                'course4': course_grades[3],
                'course5': course_grades[4],
                'average_grade': average_grade,
                'passes': passes
            }

            student_data.append(student_record)

        df = pd.DataFrame(student_data)
        df.to_csv('underperform_students.csv', index=False)
        self.stdout.write(self.style.SUCCESS(f'Successfully exported {len(student_data)} student records to CSV'))