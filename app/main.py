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

# ✅ Add this Pydantic model to define expected JSON input
class FileRequest(BaseModel):
    file_url: str

@app.get("/")
def health_check():
    return {"status": "ok"}

# ✅ Updated to return only top 3 rows with specific 4 columns
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

        # Score
        result_df = score_uploaded_file(df)

        # ✅ Only keep top 3 rows and required columns
        top3 = result_df.head(3)

        # ✅ Build minimal response with 4 specific fields
        output = {
            "Rows": {
                "Rows Row Date": top3["Date"].astype(str).tolist(),
                "Rows Row Day": top3["Day"].astype(str).tolist(),
                "Rows Row Source": top3["Source"].astype(str).tolist(),
                "Rows Row Final Score": top3["Final_Score"].astype(str).tolist()
            }
        }

        return JSONResponse(content=output)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Ensure Render-compatible port binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
