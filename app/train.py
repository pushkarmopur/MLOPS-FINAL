import pandas as pd
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def train_model():
    print("Fetching and preprocessing data...")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')

    data = data.fillna(data.mean())
    data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)

    X = data.drop('quality', axis=1)
    y = data['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ))
    ])

    mlflow.set_experiment("Wine_Quality_Assessment")

    with mlflow.start_run():
        print("Training pipeline...")

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        new_accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", new_accuracy)
        mlflow.sklearn.log_model(pipeline, "model")

        ACCURACY_THRESHOLD = 0.75

        if new_accuracy >= ACCURACY_THRESHOLD:
            print("Model meets threshold. Saving model.pkl!")

            with open('model.pkl', 'wb') as f:
                pickle.dump(pipeline, f)


if __name__ == "__main__":
    train_model()
