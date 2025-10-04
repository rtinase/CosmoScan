from typing import Union
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as pyplot
import pickle

def save_model(model, scaler, features):
    """Зберігає натреновану модель та пов'язані з нею об'єкти у файли"""
    
    # Зберігаємо модель
    with open('exoplanet_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Зберігаємо scaler (важливо для правильного масштабування нових даних)
    with open('exoplanet_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Зберігаємо список ознак
    with open('exoplanet_features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("Модель, скейлер та список ознак успішно збережено!")

def predict_on_new_data(file_path):
    """Завантажує нові дані і робить прогнози, використовуючи збережену модель"""
    
    # Завантажуємо модель, скейлер і список ознак
    with open('exoplanet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('exoplanet_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('exoplanet_features.pkl', 'rb') as f:
        features = pickle.load(f)
    
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
def save_predictions(predictions, output_file='prediction_results.csv'):
    """Зберігає результати прогнозування у файл і показує статистику"""
    
    # Зберігаємо в CSV
    predictions.to_csv(output_file, index=False)
    print(f"Прогнози збережено у файлі {output_file}")
    
    # Створюємо звіт про результати
    if 'actual_class' in predictions.columns:
        # Обчислюємо точність, якщо доступні справжні класи
        accuracy = (predictions['actual_class'] == predictions['predicted_class']).mean()
        print(f"Точність прогнозу на нових даних: {accuracy:.4f} (або {accuracy*100:.2f}%)")
        
        # Створюємо матрицю плутанини
        cm = confusion_matrix(predictions['actual_class'], predictions['predicted_class'])
        
        pyplot.figure(figsize=(8, 6))
        pyplot.imshow(cm, interpolation='nearest', cmap=pyplot.cm.Blues)
        pyplot.title('Матриця плутанини для нових даних')
        pyplot.colorbar()
        pyplot.xticks([0, 1], ['Не екзопланета', 'Екзопланета'])
        pyplot.yticks([0, 1], ['Не екзопланета', 'Екзопланета'])
        pyplot.ylabel('Справжній клас')
        pyplot.xlabel('Передбачений клас')
        pyplot.tight_layout()
        pyplot.savefig('нові_дані_матриця_плутанини.png')
        print("Матрицю плутанини збережено як 'нові_дані_матриця_плутанини.png'")
    
    # Зведення прогнозів
    class_counts = predictions['predicted_class'].value_counts()
    print("\nЗведення прогнозів:")
    print(f"Прогнозовано не екзопланет (0): {class_counts.get(0, 0)}")
    print(f"Прогнозовано екзопланет (1): {class_counts.get(1, 0)}")

def load_data(file_path: str, sign: Union["comma", "semicolon"]) -> pandas.DataFrame:
    print("Start reading data...\n")
    if sign == "comma":
        dataFrame = pandas.read_csv(file_path, comment='#')
    elif sign == "semicolon":
        dataFrame = pandas.read_csv(file_path, sep=';', comment='#')
    else:
        raise ValueError("Unknown delimiter")

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

# Крок 5: Оцінка моделі
def evaluate_model(model, X_test, y_test, features):
    """
    Оцінює продуктивність моделі.
    """
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
    
    # Візуалізація матриці плутанини
    cm = confusion_matrix(y_test, y_pred)
    pyplot.figure(figsize=(8, 6))
    pyplot.imshow(cm, interpolation='nearest', cmap=pyplot.cm.Blues)
    pyplot.title('Матриця плутанини')
    pyplot.colorbar()
    pyplot.xticks([0, 1], ['Не екзопланета', 'Екзопланета'])
    pyplot.yticks([0, 1], ['Не екзопланета', 'Екзопланета'])
    
    pyplot.ylabel('Справжній клас')
    pyplot.xlabel('Передбачений клас')
    pyplot.tight_layout()
    pyplot.savefig('матриця_плутанини.png')
    
    return y_pred, accuracy

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Інструмент класифікації екзопланет')
    parser.add_argument('--train', action='store_true', help='Навчити нову модель')
    parser.add_argument('--predict', type=str, help='Шлях до нових даних для прогнозування')
    parser.add_argument('--output', type=str, default='prediction_results.csv', 
                        help='Вихідний файл для результатів прогнозування')
    
    args = parser.parse_args()
    
    if args.train:
        # Навчання нової моделі
        df = load_data("./kepler_objects_of_interest.csv", "comma")
        df_processed, features = preprocess_data(df)
        X_train, X_test, y_train, y_test, scaler, features = prepare_training_data(df_processed, features)
        model = train_model(X_train, y_train)
        y_pred, accuracy = evaluate_model(model, X_test, y_test, features)
        
        # Зберігаємо модель для подальшого використання
        save_model(model, scaler, features)
        
        print("\n====== RESULT ======")
        print(f"Accuracy: {accuracy*100:.2f}%.")
        print("Confusion matrix image saved as 'confusion_matrix.png'.")
        print("Модель збережено для майбутніх прогнозів.")
    
    elif args.predict:
        # Прогнозування на нових даних
        predictions = predict_on_new_data(args.predict)
        if predictions is not None:
            save_predictions(predictions, args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
    