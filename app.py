from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, storage, firestore
from deepface import DeepFace
import os
import gdown

# Initialize Firebase
cred = credentials.Certificate(r"flook-beta-c1d1d-firebase-adminsdk-6gam1-071cdddfa9.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'flook-beta-c1d1d.appspot.com'})
db = firestore.client()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/upload_images')
async def upload_images(data: dict):
    drive_folder_link = data['drive_folder_link']
    user_email = data['email']
    album_id = data['album_id']
    folder_id = drive_folder_link.split('/')[-1].split('?')[0]

    # Define directory to save files temporarily
    download_dir = os.path.join('folders', folder_id)
    os.makedirs(download_dir, exist_ok=True)

    # Download files from Google Drive
    try:
        gdown.download_folder(drive_folder_link, output=download_dir, quiet=False)
    except Exception as e:
        return JSONResponse(status_code=400, content={'error': str(e)})

    try:
        dfs = DeepFace.find(img_path="example.png", db_path=download_dir, model_name='Facenet')
    except Exception as e:
        print("PKL Created. No face found")

    try:
        album_ref = db.collection('albums').document(album_id)
        album_ref.update({'status': 'ready'})
        print(f"Album {album_id} status updated to 'ready'.")
    except Exception as e:
        return JSONResponse(status_code=400, content={'error': f"Could not update album status: {str(e)}"})

    return JSONResponse(content={'message': 'Files uploaded successfully!'})

@app.post('/search_photos')
async def search_photos(file: UploadFile = File(...), album_id: str = Form(...)):
    if not file or file.filename == '':
        raise HTTPException(status_code=400, detail="No file selected")

    # Create the 'faces' directory if it doesn't exist
    faces_dir = 'faces'
    os.makedirs(faces_dir, exist_ok=True)

    # Save the face image in the 'faces' directory
    save_path = os.path.join(faces_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Construct the download directory path based on album_id
    download_dir = os.path.join('folders', album_id)

    # Use DeepFace to search for matches
    try:
        dfs = DeepFace.find(img_path=save_path, db_path=download_dir, model_name='Facenet')
        df = dfs[0]
        res = df['identity'].values.tolist()
        urls = [f'http://127.0.0.1:8000/images/{album_id}/' + os.path.basename(file) for file in res]
        return JSONResponse(content=urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/get_photos')
async def get_photos(data: dict):
    album_id = data['album_id'].split('/')[1]
    download_dir = os.path.join('', album_id)
    urls = [f'http://127.0.0.1:8000/images/{album_id}/' + file for file in os.listdir("folders/" + download_dir) if not file.endswith('.pkl')]
    return JSONResponse(content=urls)

@app.get('/hello')
async def hello_world():
    return JSONResponse(content={"message": "Hello, Flook!"})

@app.get('/images/{driveId}/{path}')
async def serve_image(driveId: str, path: str):
    album_name = os.listdir('folders/' + driveId)[0]
    thumbnail = driveId + '/' + path
    return FileResponse(os.path.join('folders', thumbnail))

# Run the app using an ASGI server like uvicorn
# Use the command: uvicorn your_module_name:app --reload
