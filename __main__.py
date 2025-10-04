# Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Крок 1: Створення вибірки даних
def load_sample_data():
    """
    Створює вибірку даних для демонстрації.
    У реальному випадку ви б завантажували дані з файлу.
    """
    print("Створення вибірки даних...")
    
    # Створюємо простий набір даних для класифікації екзопланет
    data = {
        'koi_disposition': ['CONFIRMED', 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE', 
                           'FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
        'koi_period': [5.7, 2.8, 4.2, 1.3, 6.5, 3.9, 7.1, 0.5],
        'koi_duration': [2.5, 1.9, 3.2, 0.8, 1.1, 2.7, 3.5, 0.3],
        'koi_depth': [0.05, 0.08, 0.03, 0.01, 0.02, 0.04, 0.09, 0.01],
        'koi_model_snr': [15.2, 20.4, 12.1, 5.6, 6.8, 11.3, 25.7, 4.2],
        'koi_prad': [2.1, 1.8, 2.5, 0.9, 1.2, 2.2, 3.0, 0.7],
        'koi_teq': [780, 1200, 650, 1500, 550, 850, 730, 1800]
    }
    
    df = pd.DataFrame(data)
    print(f"Створено датасет з {len(df)} рядків та {len(df.columns)} стовпців.")
    
    return df

# Крок 2: Попередня обробка даних
def preprocess_data(df):
    """
    Обробляє дані для підготовки їх до навчання моделі.
    """
    print("\nПопередня обробка даних...")
    
    # Створюємо цільову змінну (1 - екзопланета, 0 - не екзопланета)
    df['target'] = df['koi_disposition'].apply(
        lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
    )
    
    # Вибираємо числові ознаки для навчання
    features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_model_snr', 'koi_prad', 'koi_teq']
    
    # Заповнюємо пропущені значення середніми значеннями
    for col in features:
        df[col] = df[col].fillna(df[col].mean())
    
    return df, features

# Крок 3: Підготовка даних для навчання
def prepare_training_data(df, features):
    """
    Готує дані для навчання моделі.
    """
    print("\nПідготовка даних для навчання...")
    
    # Підготовка ознак та цільової змінної
    X = df[features]
    y = df['target']
    
    # Масштабування ознак
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Розділення даних на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, features

# Крок 4: Навчання моделі
def train_model(X_train, y_train):
    """
    Навчає модель машинного навчання.
    """
    print("\nНавчання моделі випадкового лісу...")
    
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
    feature_importance = pd.DataFrame({
        'Ознака': features,
        'Важливість': importances
    }).sort_values(by='Важливість', ascending=False)
    print("\nВажливість ознак:")
    print(feature_importance)
    
    # Візуалізація матриці плутанини
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матриця плутанини')
    plt.colorbar()
    plt.xticks([0, 1], ['Не екзопланета', 'Екзопланета'])
    plt.yticks([0, 1], ['Не екзопланета', 'Екзопланета'])
    
    plt.ylabel('Справжній клас')
    plt.xlabel('Передбачений клас')
    plt.tight_layout()
    plt.savefig('матриця_плутанини.png')
    
    return y_pred, accuracy

# Головна функція
def main():
    """
    Головна функція, яка виконує всі кроки процесу машинного навчання.
    """
    print("====== МОДЕЛЬ КЛАСИФІКАЦІЇ ЕКЗОПЛАНЕТ ======\n")
    
    # Крок 1: Завантаження даних
    df = load_sample_data()
    
    # Крок 2: Попередня обробка даних
    df_processed, features = preprocess_data(df)
    
    # Крок 3: Підготовка даних для навчання
    X_train, X_test, y_train, y_test, scaler, features = prepare_training_data(df_processed, features)
    
    # Крок 4: Навчання моделі
    model = train_model(X_train, y_train)
    
    # Крок 5: Оцінка моделі
    y_pred, accuracy = evaluate_model(model, X_test, y_test, features)
    
    print("\n====== ВИСНОВОК ======")
    print(f"Модель класифікації екзопланет досягла точності {accuracy*100:.2f}%.")
    print("Зображення матриці плутанини збережено у файлі 'матриця_плутанини.png'.")

# Запускаємо програму
if __name__ == "__main__":
    main()