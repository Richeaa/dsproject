# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has on_delete set to the desired behavior
#   * Remove managed = False lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class ActivityType(models.Model):
    type_id = models.AutoField(primary_key=True)
    type_name = models.CharField(max_length=50)

    class Meta:
        managed = False
        db_table = 'activity_type'


class Course(models.Model):
    course_id = models.AutoField(primary_key=True)
    course_name = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'course'


class CourseActivity(models.Model):
    activity_id = models.AutoField(primary_key=True)
    type = models.ForeignKey(ActivityType, models.DO_NOTHING, blank=True, null=True)
    course = models.ForeignKey(Course, models.DO_NOTHING, blank=True, null=True)
    activity_name = models.CharField(max_length=100, blank=True, null=True)
    activity_start_date = models.DateTimeField(blank=True, null=True)
    activity_end_date = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'course_activity'


class Enrollment(models.Model):
    enroll_id = models.AutoField(primary_key=True)
    stu = models.ForeignKey('Student', models.DO_NOTHING, blank=True, null=True)
    course = models.ForeignKey(Course, models.DO_NOTHING, blank=True, null=True)
    grade = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'enrollment'


class Student(models.Model):
    stu_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)
    dob = models.DateField()

    class Meta:
        managed = False
        db_table = 'student'


class StudentActivityLog(models.Model):
    stu = models.ForeignKey(Student, models.DO_NOTHING)
    activity = models.ForeignKey(CourseActivity, models.DO_NOTHING)
    activity_start = models.DateTimeField()
    activity_end = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'student_activity_log'


class ModelInfo3(models.Model):
    model_name = models.CharField(max_length=100)
    model_file = models.CharField(max_length=255)
    training_data = models.CharField(max_length=255)
    training_date = models.DateTimeField()
    model_summary = models.TextField(blank=True)

    def __str__(self):
        return f"{self.model_name} - {self.training_date.strftime('%Y-%m-%d')}"