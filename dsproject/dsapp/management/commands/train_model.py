from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

from dsapp.models import ModelInfo3  # pastikan ModelInfo3 sudah di-import

class Command(BaseCommand):
    help = 'Train clustering model for LMS student activity'

    def handle(self, *args, **kwargs):
        # Load dataset
        df = pd.read_csv('grade_dataset.csv')

        features = ['total_study_time', 'avg_study_time', 'activity_count', 'active_days']
        X = df[features]

        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster = kmeans.fit_predict(X_scaled)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_.sum()

        # Save models
        model_filename = 'kmeans_model.pkl'
        scaler_filename = 'scaler.pkl'
        pca_filename = 'pca.pkl'

        joblib.dump(kmeans, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(pca, pca_filename)

        self.stdout.write(self.style.SUCCESS('✅ Model, scaler, and PCA saved'))

        # Save model info to database
        summary = f"KMeans clustering with 2 clusters.\nPCA explained variance: {explained_variance:.2%}"

        model_info = ModelInfo3.objects.create(
            model_name='KMeans Clustering (2 Clusters)',
            model_file=model_filename,
            training_data='grade_dataset.csv',
            training_date=pd.Timestamp.now(tz='Asia/Jakarta'),
            model_summary=summary,
            creator='Kaffah',
            usecase='Cluster students based on LMS study behavior to support adaptive teaching strategies.'
        )

        self.stdout.write(self.style.SUCCESS(f'Model info saved to database: ID {model_info.id}'))
