from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import requests
from io import StringIO
from app.model_logic import score_uploaded_file
import os
import uvicorn

app = FastAPI()

# ✅ Define expected request body
class FileRequest(BaseModel):
    file_url: str

@app.get("/")
def health_check():
    return {"status": "ok"}

# ✅ Prediction endpoint: now returns top 50 rows with all columns
@app.post("/predict/")
async def predict(request_data: FileRequest):
    try:
        file_url = request_data.file_url
        print("✅ Parsed file_url:", file_url)

        # Fetch file
        response = requests.get(file_url)
        if response.status_code != 200:
            return JSONResponse(status_code=400, content={"error": f"Failed to fetch file: {response.status_code}"})

        df = pd.read_csv(StringIO(response.content.decode("utf-8")))

        # Validate required columns
        required_cols = ["Description", "Reference", "Net", "Date", "GL Account Category"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return JSONResponse(status_code=400, content={"error": f"Missing required columns: {missing}"})

        # Score the uploaded file
        result_df = score_uploaded_file(df)

        # ✅ Keep top 50 rows
        top_50 = result_df.head(50)

        # ✅ Build output dynamically for all columns
        output = {f"Rows Row {col}": top_50[col].astype(str).tolist() for col in top_50.columns}

        return JSONResponse(content={"Rows": output})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Render-compatible port binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
