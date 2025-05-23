from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from .models import ModelInfo3
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

def retrain_model_view(request, model_id):
    model_info = get_object_or_404(ModelInfo3, id=model_id)
    model_name = model_info.model_name.lower()

    try:
        if model_name in ['random forest']:
          
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
        
            report = classification_report(y_test, y_pred)
            
            model_path = f'{model_name.replace(" ", "_")}_retrained.pkl'
            joblib.dump(model, model_path)

           
            model_info.training_data = 'activity_logs_detailed.csv'
            model_info.training_date = pd.Timestamp.now(tz='Asia/Jakarta')
            model_info.model_summary = report
            model_info.model_file = model_path
            model_info.creator = 'Rich'
            model_info.usecase = 'Predict the most effective times to schedule student activities by analyzing time-based patterns and activity types'
            model_info.save()

        elif model_name in ['kmeans clustering (2 clusters)']:
            
            df = pd.read_csv('grade_dataset.csv')
            features = ['total_study_time', 'avg_study_time', 'activity_count', 'active_days']
            X = df[features]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(X_scaled)

            pca = PCA(n_components=2)
            pca.fit(X_scaled)

           
            model_path = f'{model_name.replace(" ", "_")}_retrained.pkl'
            scaler_path = 'scaler_retrained.pkl'
            pca_path = 'pca_retrained.pkl'

            joblib.dump(kmeans, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(pca, pca_path)

            
            model_info.training_data = 'grade_dataset.csv'
            model_info.training_date = pd.Timestamp.now(tz='Asia/Jakarta')
            model_info.model_summary = f'KMeans with 2 clusters. PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}'
            model_info.model_file = model_path
            model_info.creator = 'Kaffah'
            model_info.usecase = 'Cluster students based on LMS activity to identify learning engagement patterns.'
            model_info.save()
        
        elif model_name in ['xgboost regressor', 'random forest regressor']:
            df = pd.read_csv('stu_activ_recom.csv').fillna(0)

            le = LabelEncoder()
            df["gender_encoded"] = le.fit_transform(df["gender"])

            df.rename(columns={
                'time_type_ActivityType object (1)': 'time_Quiz',
                'time_type_ActivityType object (2)': 'time_IndividualAssignment',
                'time_type_ActivityType object (3)': 'time_GroupAssignment',
                'time_type_ActivityType object (4)': 'time_Forum'
            }, inplace=True)

            activity_cols = ['time_Quiz', 'time_IndividualAssignment', 'time_GroupAssignment', 'time_Forum']
            features = ["age", "gender_encoded"] + activity_cols
            X = df[features]
            y = df["avg_grade"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_name == 'xgboost regressor':
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
            elif model_name == 'random forest regressor':
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            model_path = f'{model_name.replace(" ", "_")}_retrained.pkl'
            joblib.dump(model, model_path)

            model_info.training_data = 'stu_activ_recom.csv'
            model_info.training_date = pd.Timestamp.now(tz='Asia/Jakarta')
            model_info.model_summary = f"Model trained with MSE: {mse:.2f}, MAE: {mae:.2f}."
            model_info.model_file = model_path
            model_info.creator = 'Aqeel'
            model_info.usecase = 'Recommend optimal learning activities based on student profiles and past performance.'
            model_info.save()


        else:
            messages.error(request, f"Unknown model name: {model_info.model_name}")
            return redirect('/admin/dsapp/modelinfo3/')

        messages.success(request, f"Model '{model_info.model_name}' retrained successfully!")

    except Exception as e:
        messages.error(request, f"Error during retraining: {str(e)}")

    return redirect('/admin/dsapp/modelinfo3/')

