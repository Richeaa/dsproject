import csv
from datetime import date
from django.core.management.base import BaseCommand
from dsapp.models import Student, Enrollment, StudentActivityLog, CourseActivity

class Command(BaseCommand):
    help = 'Export data for clustering model (with student name)'

    def handle(self, *args, **kwargs):
        output_file = 'grade_dataset.csv'

        # Buat dictionary: {stu_id (int): Student instance}
        students = {s.stu_id: s for s in Student.objects.all()}
        enrollments = Enrollment.objects.all()
        logs = StudentActivityLog.objects.values(
            'stu_id', 'activity_id', 'activity_start', 'activity_end'
        ).iterator()
        course_activities = {c.activity_id: c.course_id for c in CourseActivity.objects.all()}

        activity_data = {}

        for log in logs:
            sid = log['stu_id']
            cid = course_activities.get(log['activity_id'])

            duration = (log['activity_end'] - log['activity_start']).total_seconds() / 60  # minutes
            date_key = log['activity_start'].date()

            if sid not in activity_data:
                activity_data[sid] = {}

            if cid not in activity_data[sid]:
                activity_data[sid][cid] = {
                    'total_time': 0,
                    'count': 0,
                    'active_days': set()
                }

            activity_data[sid][cid]['total_time'] += duration
            activity_data[sid][cid]['count'] += 1
            activity_data[sid][cid]['active_days'].add(date_key)

        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'stu_id', 'name', 'gender', 'age', 'course_id',
                'total_study_time', 'avg_study_time', 'activity_count', 'active_days', 'grade'
            ])

            for enr in enrollments:
                # FIX: Ambil ID-nya (int), lalu cari student dari dict
                stu_id = enr.stu_id.stu_id if hasattr(enr.stu_id, 'stu_id') else enr.stu_id
                student = students.get(stu_id)
                if not student:
                    continue

                today = date(2025, 5, 1)
                age = today.year - student.dob.year - (
                    (today.month, today.day) < (student.dob.month, student.dob.day)
                )

                act = activity_data.get(student.stu_id, {}).get(enr.course_id, {
                    'total_time': 0,
                    'count': 0,
                    'active_days': set()
                })

                total_time = act['total_time']
                count = act['count']
                avg_time = total_time / count if count else 0
                active_days = len(act['active_days'])

                writer.writerow([
                    student.stu_id,
                    student.name,
                    student.gender,
                    age,
                    enr.course_id,
                    round(total_time, 2),
                    round(avg_time, 2),
                    count,
                    active_days,
                    enr.grade
                ])

        self.stdout.write(self.style.SUCCESS(f'✅ Successfully exported dataset to {output_file}'))
