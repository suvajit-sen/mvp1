from fastapi import FastAPI, File, UploadFile, Depends
import os
import pandas as pd


app =FastAPI()



UPLOAD_DIR = "uploaded_files/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_file(file:UploadFile=File(...)):
    print(file.filename)
    file_location = f'{UPLOAD_DIR}+{file.filename}'
    with open(file_location, 'wb') as f:
        f.write(file.file.read())

    df = pd.read_excel(file_location)
    column_names = ", ".join(df.columns.tolist())    

    return {"filename":file.filename, "columns": column_names}





