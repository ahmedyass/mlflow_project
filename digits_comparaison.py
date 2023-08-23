import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

mlflow.set_experiment("Comparaison_Classification_Digits")

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net"]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier()]

# Hyperparamètres à ajuster pour chaque classificateur
param_grids = {
    "Nearest Neighbors": {'n_neighbors': [3, 5, 7, 9]},
    "Linear SVM": {'C': [0.1, 1, 10]},
    "RBF SVM": {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]},
    "Decision Tree": {'max_depth': [None, 10, 20, 30]},
    "Random Forest": {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]},
    "Neural Net": {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}
}

# Charger et normaliser les données
digits = load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for name, classifier in zip(names, classifiers):
    with mlflow.start_run(run_name=name):
        # Ajustement des hyperparamètres
        clf = GridSearchCV(classifier, param_grids[name], cv=5)
        clf.fit(X_train, y_train)
        
        # Meilleur modèle après ajustement
        best_model = clf.best_estimator_

        # Validation croisée
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
        mean_cv_score = np.mean(cv_scores)
        mlflow.log_metric("mean_cv_score", mean_cv_score)

        # Prédiction
        y_pred = best_model.predict(X_test)

        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log dans MLFlow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Log des hyperparamètres optimaux
        mlflow.log_params(clf.best_params_)

        # Sauvegarde du modèle
        mlflow.sklearn.log_model(best_model, name)
        
        # Visualisation
        plt.figure()
        plt.title(f"{name} - Confusion Matrix")
        plt.imshow(np.corrcoef(y_test.T, y_pred.T), cmap='viridis', origin='lower')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plot_path = f"{name}_confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
