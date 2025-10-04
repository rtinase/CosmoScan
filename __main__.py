# Імпорт необхідних бібліотек
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as pyplot

# Крок 1: Створення вибірки даних
def load_data():
    print("Start reading data...")

    file_path = "./kepler_objects_of_interest.csv"
    dataFrame = pandas.read_csv(file_path, comment='#')

    print(dataFrame.head()) # just display first 5 rows

    print(f"Loaded dataset with {len(dataFrame)} rows and {len(dataFrame.columns)} columns.")
    return dataFrame


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

# Головна функція
def main():
    print("====== model of classification exoplanets ======\n")
    
    df = load_data()
    
    # Крок 2: Попередня обробка даних
    # df_processed, features = preprocess_data(df)
    
    # # Крок 3: Підготовка даних для навчання
    # X_train, X_test, y_train, y_test, scaler, features = prepare_training_data(df_processed, features)
    
    # # Крок 4: Навчання моделі
    # model = train_model(X_train, y_train)
    
    # # Крок 5: Оцінка моделі
    # y_pred, accuracy = evaluate_model(model, X_test, y_test, features)
    
    # print("\n====== ВИСНОВОК ======")
    # print(f"Модель класифікації екзопланет досягла точності {accuracy*100:.2f}%.")
    # print("Зображення матриці плутанини збережено у файлі 'матриця_плутанини.png'.")

# Запускаємо програму
if __name__ == "__main__":
    main()