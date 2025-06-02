import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='rnjt80', repo_name='mlflow-practical', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/rnjt80/mlflow-practical.mlflow")

# Load Wine Dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#Define the params for RF model
max_depth = 20
n_estimators = 10

# Mention the name of your experiment below
mlflow.autolog()
mlflow.set_experiment('exp2-remote')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    #Creating a confustion metrix plot

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap ='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
   
    # Save Plot
    plt.savefig("Confusion-matrix.png")

    mlflow.log_metric('accuracy', accuracy)
    
    # Log artifacts using mlflow
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Ranjit', "Project": 'Wine Classification'})

    print(accuracy)