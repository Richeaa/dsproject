# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=False
#   * Make sure each ForeignKey and OneToOneField has on_delete set to the desired behavior
#   * Remove managed = False lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models

#Create your models here.
class Student(models.Model):
    stu_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.EmailField()
    gender = models.CharField(max_length=10)
    dob = models.DateField()

    class Meta:
        db_table = 'student'
        managed = False

class Course(models.Model):
    course_id = models.AutoField(primary_key=True)
    course_name = models.CharField(max_length=100)

    class Meta:
        db_table = 'course'
        managed = False

class Enrollment(models.Model):
    enroll_id = models.AutoField(primary_key=True)
    stu_id = models.ForeignKey(Student, db_column='stu_id', on_delete=models.CASCADE)
    course_id = models.ForeignKey(Course, db_column='course_id', on_delete=models.CASCADE)
    grade = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = 'enrollment'
        managed = False

class ActivityType(models.Model):
    type_id = models.AutoField(primary_key=True)
    type_name = models.CharField(max_length=100)

    class Meta:
        db_table = 'activity_type'
        managed = False

class CourseActivity(models.Model):
    activity_id = models.AutoField(primary_key=True)
    type_id = models.ForeignKey(ActivityType, db_column='type_id', on_delete=models.CASCADE)
    course_id = models.ForeignKey(Course, db_column='course_id', on_delete=models.CASCADE)
    activity_name = models.CharField(max_length=100)
    activity_start_date = models.DateField()
    activity_end_date = models.DateField()

    class Meta:
        db_table = 'course_activity'
        managed = False

class StudentActivityLog(models.Model):
    stu_id = models.ForeignKey(Student, db_column='stu_id', on_delete=models.CASCADE)
    activity_id = models.ForeignKey(CourseActivity, db_column='activity_id', on_delete=models.CASCADE)
    activity_start = models.DateTimeField()
    activity_end = models.DateTimeField()

    class Meta:
        db_table = 'student_activity_log'
        managed = False
        unique_together = (('stu_id', 'activity_id', 'activity_start'),)
        
class ModelInfo3(models.Model):
    model_name = models.CharField(max_length=100)
    model_file = models.CharField(max_length=255)  # you can use FileField if you're uploading .pkl files
    training_data = models.CharField(max_length=255)
    training_date = models.DateTimeField(auto_now_add=True)
    model_summary = models.TextField()
    creator = models.TextField()
    usecase = models.TextField()
    
    def __str__(self):
        return f"{self.model_name} - {self.training_date.strftime('%Y-%m-%d')}"

    class Meta:
        managed = True


class PredictionResult(models.Model):
    student_id = models.CharField(max_length=20)
    total_study_time = models.FloatField()
    avg_study_time = models.FloatField()
    activity_count = models.FloatField()
    active_days = models.FloatField()
    cluster = models.IntegerField()
    pca_x = models.FloatField()
    pca_y = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Cluster {self.cluster} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"
