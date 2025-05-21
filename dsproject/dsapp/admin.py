from django.contrib import admin
from .models import ModelInfo3
from django.utils.html import format_html

# Register your models here.
@admin.register(ModelInfo3)
class ModelInfo3Admin(admin.ModelAdmin):
    list_display =('model_name', 'training_date', 'training_data', 'model_summary', 'creator', 'usecase', 'retrain_button')

    def short_summary(self, obj):
        return (obj.model_summary[:75] + '...') if obj.model_summary else '-'
    short_summary.short_description = 'Model Summary'

    def retrain_button(self, obj):
        return format_html('<a class="button" href="/retrain-model/{}/">Retrain</a>', obj.id)
    retrain_button.short_description = 'Retrain Model'
