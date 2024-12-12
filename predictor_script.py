import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the data
data = pd.read_csv("amby_full_thread.csv")  # Replace with your dataset
data['timestamp'] = pd.to_datetime(data['timestamp'])  # Ensure timestamp is datetime
data['date'] = data['timestamp'].dt.date  # Extract date for grouping

# Group by date
data = data.groupby('date').size().reset_index(name='message_count')  # Group by day

# Ensure `date` is in datetime64 format for `.dt` operations
data['date'] = pd.to_datetime(data['date'])

# Step 2: Create labels for weekly predictions
data['next_week_message'] = data['message_count'].rolling(7).sum().shift(-7).apply(lambda x: 1 if x > 0 else 0)
data['most_likely_day'] = data['date'].shift(-1).dt.dayofweek  # Use the next message day for prediction

# Step 3: Feature engineering
today = pd.to_datetime(datetime.now().date())  # Ensure `today` is a datetime object
data['days_since_last_message'] = (today - data['date']).dt.days  # Days since last message
data['day_of_week'] = data['date'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)
data['rolling_7_day_avg'] = data['message_count'].rolling(7).mean()  # 7-day rolling average
data['rolling_7_day_sum'] = data['message_count'].rolling(7).sum()  # 7-day rolling sum
data['rolling_3_day_avg'] = data['message_count'].rolling(3).mean()  # 3-day rolling average
data['previous_day_count'] = data['message_count'].shift(1)  # Previous day's count

# Drop NaNs created by rolling and shifting
data = data.dropna()

# Step 4: Define features and target for week prediction
week_features = ['day_of_week', 'rolling_7_day_avg', 'rolling_7_day_sum', 'rolling_3_day_avg', 
                 'previous_day_count', 'days_since_last_message']
week_target = 'next_week_message'

X_week = data[week_features]
y_week = data[week_target]

# Step 5: Train-test split for week prediction
X_week_train, X_week_test, y_week_train, y_week_test = train_test_split(X_week, y_week, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest Classifier for week prediction
week_model = RandomForestClassifier(random_state=42)
week_model.fit(X_week_train, y_week_train)

# Step 7: Evaluate the week prediction model
y_week_pred = week_model.predict(X_week_test)
print("Week Prediction Accuracy:", accuracy_score(y_week_test, y_week_pred))
print("Week Prediction Classification Report:\n", classification_report(y_week_test, y_week_pred))

# Feature importance for week prediction
week_feature_importance = pd.DataFrame({'Feature': week_features, 'Importance': week_model.feature_importances_})
print("Week Prediction Feature Importance:\n", week_feature_importance.sort_values(by='Importance', ascending=False))

# Step 8: Train a model for day prediction if a message is likely in the following week
day_data = data[data['next_week_message'] == 1]  # Filter only rows where a message is expected
day_features = ['day_of_week', 'rolling_7_day_avg', 'rolling_7_day_sum', 'rolling_3_day_avg', 
                'previous_day_count', 'days_since_last_message']
day_target = 'most_likely_day'

X_day = day_data[day_features]
y_day = day_data[day_target]

# Train-test split for day prediction
X_day_train, X_day_test, y_day_train, y_day_test = train_test_split(X_day, y_day, test_size=0.2, random_state=42)

# Train a Random Forest Classifier for day prediction
day_model = RandomForestClassifier(random_state=42)
day_model.fit(X_day_train, y_day_train)

# Evaluate the day prediction model
y_day_pred = day_model.predict(X_day_test)
print("Day Prediction Accuracy:", accuracy_score(y_day_test, y_day_pred))
print("Day Prediction Classification Report:\n", classification_report(y_day_test, y_day_pred))

# Feature importance for day prediction
day_feature_importance = pd.DataFrame({'Feature': day_features, 'Importance': day_model.feature_importances_})
print("Day Prediction Feature Importance:\n", day_feature_importance.sort_values(by='Importance', ascending=False))

# Step 9: Make a prediction for the following week
def predict_next_week_and_day(latest_data):
    week_prediction = week_model.predict(latest_data[week_features].tail(1))
    if week_prediction[0] == 1:  # If a message is predicted
        day_prediction = day_model.predict(latest_data[day_features].tail(1))
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(f"Prediction: Yes, a message is likely next week. Most likely day: {days[day_prediction[0]]}.")
    else:
        print("Prediction: No, a message is unlikely next week.")

# Use the most recent data row for prediction
predict_next_week_and_day(data)