from django.core.management.base import BaseCommand
import pandas as pd
from dsapp.models import StudentActivityLog
from django.utils import timezone
from datetime import timezone as dt_timezone
from collections import defaultdict

class Command(BaseCommand):
    help = "ETL: Extract detailed activity data for ML prediction with extended features"

    def handle(self, *args, **kwargs):
        print("Extracting detailed activity data for prediction...")

        logs = StudentActivityLog.objects.select_related(
            'activity_id__type_id', 'stu_id'
        ).filter(
            activity_end__isnull=False
        )

        ml_data = []
        activity_counts = defaultdict(int)
        processed = 0

        for log in logs:
            try:
                if not all([log.activity_start, log.activity_end, log.activity_id, log.activity_id.type_id, log.stu_id]):
                    continue

                start_dt = log.activity_start
                end_dt = log.activity_end

                if timezone.is_naive(start_dt):
                    start_dt = timezone.make_aware(start_dt, dt_timezone.utc)
                if timezone.is_naive(end_dt):
                    end_dt = timezone.make_aware(end_dt, dt_timezone.utc)

                local_start = timezone.localtime(start_dt)
                local_end = timezone.localtime(end_dt)

                duration = (local_end - local_start).total_seconds() / 60
                if duration <= 0 or duration > 480:
                    continue

                day = local_start.strftime('%A')
                hour = local_start.hour
                minute = local_start.minute
                day_of_week = local_start.weekday() + 1
                type_name = log.activity_id.type_id.type_name
                minutes_since_midnight = hour * 60 + minute

                # Durasi kategori
                if duration <= 10:
                    duration_cat = "short"
                elif duration <= 30:
                    duration_cat = "medium"
                else:
                    duration_cat = "long"

                # Hitung aktivitas harian per mahasiswa
                date_key = local_start.date()
                student_key = (log.stu_id.stu_id, date_key)
                activity_counts[student_key] += 1
                daily_activity_count = activity_counts[student_key]

                # Kombinasi peak dan weekend
                is_peak_or_weekend = 1 if hour in [19, 20, 21] or day in ['Saturday', 'Sunday'] else 0

                record = {
                    'student_id': log.stu_id.stu_id,
                    'type_name': type_name,
                    'day': day,
                    'hour': hour,
                    'minute': minute,
                    'day_of_week': day_of_week,
                    'duration': round(duration, 2),
                    'is_weekend': 1 if day in ['Saturday', 'Sunday'] else 0,
                    'is_morning': 1 if 6 <= hour <= 11 else 0,
                    'is_afternoon': 1 if 12 <= hour <= 17 else 0,
                    'is_evening': 1 if 18 <= hour <= 22 else 0,
                    'is_peak': 1 if hour in [19, 20, 21] else 0,
                    'minutes_since_midnight': minutes_since_midnight,
                    'duration_category': duration_cat,
                    'is_early_morning': 1 if 0 <= hour < 6 else 0,
                    'is_peak_or_weekend': is_peak_or_weekend,
                    'daily_activity_count': daily_activity_count
                }

                ml_data.append(record)
                processed += 1
                if processed % 1000 == 0:
                    print(f"Processed {processed} logs...")

            except Exception as e:
                continue

        df = pd.DataFrame(ml_data)
        df.to_csv('activity_logs_detailed.csv', index=False)

        print(f"Saved {len(df)} rows to 'activity_logs_detailed.csv'")
