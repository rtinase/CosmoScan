from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import os
from ml import predict
from help_functions import get_avaliable_cols_from

app = FastAPI(title="Exoplanet Classification API",
              description="API for expoplanet finding from Kepler Data")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        predictions = predict(temp_file_path)
        
        if predictions is None:
            raise HTTPException(status_code=500, 
                               detail="Did not manage to make predictions. Check server logs.")
        
        result_path = f"{temp_file_path}_results.csv"

        available_cols = get_avaliable_cols_from(predictions, "api")

        
            
        result_df = predictions[available_cols].copy()
        result_df.to_csv(result_path, index=False)
        
        return FileResponse(
            result_path,
            media_type="text/csv",
            filename="prediction_results.csv"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        
    finally:
        if 'temp_file_path' in locals():
            try:
                os.remove(temp_file_path)
            except:
                pass


@app.get("/health")
async def health_check():
    if model_files_exist():
        return {"status": "ok"}
    else:
        return {"status": "error", "detail": "Model files are missing"}


def model_files_exist() -> bool:
    required_files = [
        './models/exoplanet_model.joblib',
        './models/exoplanet_scaler.joblib',
        './models/exoplanet_features.joblib'
    ]
    return all(os.path.isfile(f) for f in required_files)