from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from io import StringIO
from app.model_logic import score_uploaded_file

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Validate required columns
        required_cols = ["Description", "Reference", "Net", "Date", "GL Account Category"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return JSONResponse(status_code=400, content={"error": f"Missing required columns: {missing}"})

        # Run scoring
        result_df = score_uploaded_file(df)

        # Convert top 50 rows to JSON for preview
        result = result_df.head(50).to_dict(orient="records")
        return {"results": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
