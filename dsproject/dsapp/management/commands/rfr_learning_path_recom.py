import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib
from dsapp.models import ModelInfo3

class Command(BaseCommand):
    help = 'Generate learning path recommendations for a student based on their activity data'

    def handle(self, *args, **kwargs):
        df = pd.read_csv('stu_activ_recom.csv').fillna(0)

        # Encode gender
        le = LabelEncoder()
        df["gender_encoded"] = le.fit_transform(df["gender"])

        # Rename columns for clarity if not already renamed
        df.rename(columns={
            'time_type_ActivityType object (1)': 'time_Quiz',
            'time_type_ActivityType object (2)': 'time_IndividualAssignment',
            'time_type_ActivityType object (3)': 'time_GroupAssignment',
            'time_type_ActivityType object (4)': 'time_Forum'
        }, inplace=True)

        activity_cols = ['time_Quiz', 'time_IndividualAssignment', 'time_GroupAssignment', 'time_Forum']
        features = ["age", "gender_encoded"] + activity_cols
        
        # Split    
        X = df[features]
        y = df["avg_grade"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict for specific student
        student_id = 5
        student = df[df["stu_id"] == student_id]
        if student.empty:
            self.stdout.write("Student ID not found in the dataset.")
            return

        base_features = student.iloc[0][features]
        base_df = pd.DataFrame([base_features.values], columns=features)

        pred_overall = "{:.2f}".format(model.predict(base_df)[0])

        activity_predictions = {}
        for col in activity_cols:
            test_row = base_features.copy()
            test_row[activity_cols] = 0
            test_row[col] = base_features[col]
            test_df = pd.DataFrame([test_row.values], columns=features)
            pred = "{:.2f}".format(model.predict(test_df)[0])
            activity_name = col.replace("time_", "")
            activity_predictions[activity_name] = pred
            
        actual_avg_grade = student.iloc[0]["avg_grade"]

        # Output
        self.stdout.write(f"\nRecommendations for Student {student_id}:")
        self.stdout.write(f"  Actual average grade: {actual_avg_grade}")
        self.stdout.write(f"  Predicted overall grade: {pred_overall}")
        self.stdout.write("  Predicted performance on activity types:")
        for k, v in activity_predictions.items():
            self.stdout.write(f"    - {k}: {v}")
            
        # Model eval
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        self.stdout.write(f"MSE: {mse:.2f}")
        self.stdout.write(f"MAE: {mae:.2f}")
        
        # APRIORI 
        # START
        # FROM
        # HERE
        
        df_binary = df[['stu_id'] + activity_cols].copy()
        for col in activity_cols:
            df_binary[col] = df_binary[col].apply(lambda x: 1 if x > 0 else 0)

        transactions = []
        for _, row in df_binary.iterrows():
            activities = [col.replace('time_', '') for col in activity_cols if row[col] == 1]
            if activities:
                transactions.append(activities)

        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        te_df = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = apriori(te_df, min_support=0.1, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        
        filtered_rules = rules[
            (rules['support'] >= 0.1) &
            (rules['lift'] > 1.04)
        ].sort_values(by=['confidence', 'lift', 'support'], ascending=False).head(5)


        self.stdout.write("\n  Recommended activity sequences:")
        if filtered_rules.empty:
            self.stdout.write("    - No strong activity sequence rules found.")
        else:
            for _, row in filtered_rules.iterrows():
                antecedents = ', '.join(row['antecedents'])
                consequents = ', '.join(row['consequents'])
                conf = round(row['confidence'], 2)
                supp = round(row['support'], 2)
                lift = round(row['lift'], 2)
                self.stdout.write(
                    f"    - If completed {antecedents} → try {consequents} (confidence: {conf}, support: {supp}, lift: {lift})"
                )
                
        # Save model
        model_filename = 'rfr_learning_path_model.pkl'
        joblib.dump(model, model_filename)
        self.stdout.write(self.style.SUCCESS(f"Model saved to {model_filename}"))
        
        model_info = ModelInfo3.objects.create(
            model_name='Random Forest Regressor',
            model_file =model_filename,
            training_data = 'stu_activ_recom.csv',
            training_date = pd.Timestamp.now(),
            model_summary = f"Model trained with MSE: {mse:.2f}, MAE: {mae:.2f}, and saved to {model_filename}.",
            creator = 'Aqeel',
            usecase = 'Recommend optimal learning activities based on student profiles and past performance.'
        )
        self.stdout.write(self.style.SUCCESS(f"Model info saved to database with ID: {model_info.id}"))