from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import os
from __main__ import predict

app = FastAPI(title="Exoplanet Classification API",
              description="API для класифікації екзопланет за даними Kepler")

@app.post("/predict/")
async def predict_exoplanet(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV files are supported only")
    
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(contents)
        
        with open(temp_file_path, 'r') as f:
            first_line = f.readline()
            delimiter = "semicolon" if ";" in first_line else "comma"
        
        predictions = predict(temp_file_path)
        
        if predictions is None:
            raise HTTPException(status_code=500, 
                               detail="Did not manage to make predictions. Check server logs.")
        
        result_path = f"{temp_file_path}_results.csv"
        
        # Фільтруємо колонки для відповіді
        cols_to_save = ['kepid', 'kepoi_name', 'kepler_name', 'predicted_class', 'prediction_probability']
        available_cols = [col for col in cols_to_save if col in predictions.columns]
        
        if not available_cols:
            raise HTTPException(status_code=500, detail="У результатах не знайдено жодної бажаної колонки")
            
        # Створюємо датафрейм тільки з бажаними колонками
        result_df = predictions[available_cols].copy()
        result_df.to_csv(result_path, index=False)
        
        # Повертаємо файл з результатами
        return FileResponse(
            result_path,
            media_type="text/csv",
            filename="prediction_results.csv"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Сталася помилка: {str(e)}")
        
    finally:
        # Прибираємо за собою тимчасові файли
        if 'temp_file_path' in locals():
            try:
                os.remove(temp_file_path)
                # Файл з результатами буде видалено після відправки відповіді
            except:
                pass
