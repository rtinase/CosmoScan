from typing import Union
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as pyplot
import argparse
from help_functions import save_model, load_model


def development_loop(X_train, y_train, X_test, y_test, features):
    """
    Цикл розробки моделі: крос-валідація, тюнинг гіперпараметрів,
    оцінка різних моделей
    """
    print("\n======= Starting Development Loop =======")
    
    # Базова оцінка з крос-валідацією
    print("Performing baseline evaluation with 5-fold cross-validation:")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(base_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Baseline cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Тюнинг гіперпараметрів
    print("\nPerforming hyperparameter tuning with GridSearchCV:")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # Використовуємо невелику підмножину параметрів для демонстрації
    # В реальному проекті можна розширити пошук
    small_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        small_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Оцінюємо найкращу модель на тестових даних
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Test accuracy with best model: {test_accuracy:.4f}")
    
    # Аналіз важливості ознак
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Виведення матриці плутанини
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['False Positive', 'Exoplanet'])
    plt.yticks([0, 1], ['False Positive', 'Exoplanet'])
    
    # Додаємо анотації з цифрами
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Precision-Recall крива
    y_scores = best_model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    
    return best_model


def predict_on_new_data(file_path) -> pandas.DataFrame:
    model, scaler, features = load_model()
    
    # Завантажуємо нові дані
    print(f"Завантаження нових даних з {file_path}...")
    new_data = load_data(file_path, "semicolon")
    
    # Обробка даних (як при тренуванні, але без створення цільової змінної)
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

def load_data(file_path: str, sign: Union["comma", "semicolon"]) -> pandas.DataFrame:
    print("PROCESS START: reading data...\n")
    if sign == "comma":
        dataFrame = pandas.read_csv(file_path, comment='#')
    elif sign == "semicolon":
        dataFrame = pandas.read_csv(file_path, sep=';', comment='#')
    else:
        raise ValueError("Unknown delimiter")
    print("PROCESS END: reading data...\n")
    return dataFrame

def preprocess_data(dataFrame: pandas.DataFrame) -> tuple[pandas.DataFrame, list[str]]:
    print("\n Start preprocessing of the data...")
    
    # Створюємо цільову змінну (1 - екзопланета, 0 - не екзопланета)
    dataFrame['target'] = dataFrame['koi_disposition'].apply(
        lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
    )
    
    # Вибираємо числові ознаки для навчання
    features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_model_snr', 'koi_prad', 'koi_teq']
    
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
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, features


def train_model(X_train, y_train) -> RandomForestClassifier:
    print("\nTrain model using Random Forest...")
    
    # Створення та навчання моделі випадкового лісу
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Навчання моделі
    model.fit(X_train, y_train)
    
    return model

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

def train():
    data_frame = load_data("./data/kepler_objects_of_interest.csv", "comma")
    data_frame_processed, features = preprocess_data(data_frame)
    X_train, X_test, y_train, y_test, scaler, features = prepare_training_data(data_frame_processed, features)
    best_model = development_loop(X_train, y_train, X_test, y_test, features)
    y_pred, accuracy = evaluate_model(best_model, X_test, y_test, features)
    save_model(best_model, scaler, features)

    print("\n====== RESULT ======")
    print(f"Accuracy: {accuracy*100:.2f}%.")


def predict(file_path):
    predictions = predict_on_new_data(file_path)

    if predictions is not None:
        save_predictions(predictions)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--predict')
    
    args = parser.parse_args()
    
    if args.train:
        train()
    elif args.predict:
        predict(args.predict)

if __name__ == "__main__":
    main()
    