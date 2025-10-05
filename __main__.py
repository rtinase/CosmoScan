from typing import Union
import pandas
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import argparse
import uvicorn
from help_functions import save_model, load_model, load_data


def train_model(X_train, y_train, X_test, y_test, features): # should be checked if this one or train method is better 
    print("START PROCESS: train model")
    base_model = RandomForestClassifier(n_estimators=100)
    cross_val_score(base_model, X_train, y_train, cv=5, scoring='accuracy')
    
    param_grid = { # take this or this parameter
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    

    small_param_grid = { # take this or this parameter
        'n_estimators': [100, 200],
        'max_depth': [None, 20]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        small_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Test accuracy with best model: {test_accuracy:.4f}")
    
    feature_importance = pandas.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return best_model


def predict(file_path) -> pandas.DataFrame:
    model, scaler, features = load_model()
    
    print(f"Завантаження нових даних з {file_path}...")
    new_data = load_data(file_path, "semicolon")
    
    for col in features:
        if col in new_data.columns:
            new_data[col] = new_data[col].fillna(new_data[col].mean())
        else:
            print(f"Увага: Ознаку '{col}' не знайдено у нових даних.")
            return None
    
    # Витягуємо ознаки
    X_new = new_data[features]
    
    # Масштабуємо ознаки за допомогою збереженого скейлера
    X_new_scaled = scaler.transform(X_new)
    
    # Робимо прогноз
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]  # Імовірність класу 1
    
    # Додаємо прогнози до вихідного датафрейму
    new_data['predicted_class'] = predictions
    new_data['prediction_probability'] = probabilities
    
    # Якщо є справжні класи, додаємо їх для порівняння
    if 'koi_disposition' in new_data.columns:
        new_data['actual_class'] = new_data['koi_disposition'].apply(
            lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
        )
    
    return new_data

def save_predictions(predictions: pandas.DataFrame):
    cols_to_save = ['kepid', 'kepler_name', 'predicted_class', 'prediction_probability']
    available_cols = [col for col in cols_to_save if col in predictions.columns]
    
    if not available_cols:
        print("Помилка: Жодна з бажаних колонок не знайдена в результатах")
        return

    for col in cols_to_save:
        if col not in available_cols:
            print(f"Інформація: Колонка '{col}' не знайдена в результатах")
    
    result_df = predictions[available_cols].copy()
    output_file='./data/prediction_results.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Прогнози збережено у файлі {output_file} з колонками: {', '.join(available_cols)}")
    
    # Створюємо звіт про результати
    if 'actual_class' in predictions.columns:
        # Обчислюємо точність, якщо доступні справжні класи
        accuracy = (predictions['actual_class'] == predictions['predicted_class']).mean()
        print(f"Точність прогнозу на нових даних: {accuracy:.4f} (або {accuracy*100:.2f}%)")
    
    # Зведення прогнозів
    class_counts = predictions['predicted_class'].value_counts()
    print("\nЗведення прогнозів:")
    print(f"Прогнозовано не екзопланет (0): {class_counts.get(0, 0)}")
    print(f"Прогнозовано екзопланет (1): {class_counts.get(1, 0)}")



def preprocess_data(dataFrame: pandas.DataFrame) -> tuple[pandas.DataFrame, list[str]]:
    print("\n Start preprocessing of the data...")
    
    # Створюємо цільову змінну (1 - екзопланета, 0 - не екзопланета)
    dataFrame['target'] = dataFrame['koi_disposition'].apply(
        lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
    )

    numeric_cols = dataFrame.select_dtypes(include=['number']).columns.tolist()
    excluded_cols = ['target', 'kepid', 'rowid']
    features = [col for col in numeric_cols if col not in excluded_cols]
    
    print(f"Selected {len(features)} numeric features for training")
    
    # Заповнюємо пропущені значення середніми значеннями
    for col in features:
        dataFrame[col] = dataFrame[col].fillna(dataFrame[col].mean())
    
    return dataFrame, features

def prepare_training_data(dataFrame: pandas.DataFrame, features: list[str]) -> tuple:
    print("\nPrepare training data...")
    
    # Підготовка ознак та цільової змінної
    X = dataFrame[features]
    y = dataFrame['target']

    # Масштабування ознак
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Розділення даних на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3
    )
    
    return X_train, X_test, y_train, y_test, scaler, features

def evaluate_model(model, X_test, y_test, features):
    print("\nОцінка моделі...")
    
    # Прогнозування на тестовому наборі
    y_pred = model.predict(X_test)
    
    # Розрахунок точності
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точність моделі: {accuracy:.4f} (або {accuracy*100:.2f}%)")
    
    # Важливість ознак
    importances = model.feature_importances_
    feature_importance = pandas.DataFrame({
        'Ознака': features,
        'Важливість': importances
    }).sort_values(by='Важливість', ascending=False)
    print("\nВажливість ознак:")
    print(feature_importance)
    
    return y_pred, accuracy

def train_process():
    data_frame = load_data("./data/kepler_objects_of_interest.csv", "comma")
    data_frame_processed, features = preprocess_data(data_frame)
    X_train, X_test, y_train, y_test, scaler, features = prepare_training_data(data_frame_processed, features)
    model = train_model(X_train, y_train, X_test, y_test, features)
    y_pred, accuracy = evaluate_model(model, X_test, y_test, features)
    save_model(model, scaler, features)

    print("\n====== RESULT ======")
    print(f"Accuracy: {accuracy*100:.2f}%.")


def predict_process(file_path):
    predictions = predict(file_path)

    if predictions is not None:
        save_predictions(predictions)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--predict')
    
    args = parser.parse_args()
    
    uvicorn.run("endpoint:app", host="0.0.0.0", port=8000)

    if args.train:
        train_process()
    elif args.predict:
        predict_process(args.predict)

if __name__ == "__main__":
    main()
    