import enum
from typing import Optional, List
from enum import Enum
import time
import uvicorn
from fastapi import FastAPI, Query, Path, Cookie, Header, File, UploadFile, Depends, Request, Response, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse
# from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
import shutil, random, json , cv2, torch
from PIL import Image
import numpy as np
from pathlib import Path
import logging
logging.basicConfig(level = logging.INFO)

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import segment_processor

# http://127.0.0.1:8000/redoc
# http://127.0.0.1:8000/docs

class Result(BaseModel):
    x: List[int]
    y: List[int]
    height: int
    width: int
    render_path: Optional[str] = None

app = FastAPI(debug=False)



TEMP_DIR = '/tmp/model/segment_schp'

@app.on_event("startup")
async def startup_event():
    logging.info('Loading model')
    global box_model;global seg_model;
    box_model,seg_model = segment_processor.load_model()
    # segment_processor.parse_image(model,'/home/deeplab/datasets/custom_fashion/demo_/1544/15445714/15445714-1.jpg')
    logging.info('Loaded model')

@app.get("/classes")
def get_classes():
    """
    Returns segmentation classes dict
    """
    return segment_processor.reduced_dataset['labels']

class BoxResult(BaseModel):
    height: int
    width: int
    bboxes: List[List[float]] = []
    condidences: List[float] = []

@app.post("/boxes", response_model=BoxResult)
def get_person_boxes(response: Response, upload_file: UploadFile = File(...)):
    """
    Returns person bounding boxes (lef,upper,right,bottom)
    """
    # Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    # new_image_file = os.path.join(TEMP_DIR, 
    #     ''.join(random.choice('abcdefgh') for i in range(4))+'_'+ upload_file.filename)
    # # https://stackoverflow.com/questions/63580229/how-to-save-uploadfile-in-fastapi
    # with open(new_image_file, "wb+") as file_object:
    #     shutil.copyfileobj(upload_file.file, file_object)
    # im = cv2.imread(upload_file.file, cv2.IMREAD_COLOR)

    input = []
    image_rgb = np.array(Image.open(upload_file.file))
    img_tensor = torch.from_numpy(image_rgb/255.).permute(2,0,1).float().to(next(box_model.parameters()).device)
    input.append(img_tensor)
    pred_boxes = segment_processor.get_person_detection_boxes(box_model, input, threshold=0.9)
    result  = BoxResult(width=image_rgb.shape[1], height=image_rgb.shape[0])
    for box in pred_boxes:
        result.bboxes.append([*box[0],*box[1]])
    return result

class DATASETS(str, Enum):
    REDUCED = 'reduced'
    CIHP = 'lip'



@app.post("/segment/")
def segment(response: Response, background_tasks: BackgroundTasks, 
        upload_file: UploadFile = File(...), 
        render:bool = False, dataset_labels: DATASETS = DATASETS.REDUCED
        ):
    """
    Segment image for one person
    """
    #TODO max dim size
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    new_image_file = os.path.join(TEMP_DIR, 
        ''.join(random.choice('abcdefgh') for i in range(4))+'_'+ upload_file.filename)
    render_file = new_image_file + '.seg_schp.render.jpg'
    segment_file = new_image_file + '.seg_schp.render.png'
    try:
        # https://stackoverflow.com/questions/63580229/how-to-save-uploadfile-in-fastapi
        with open(new_image_file, "wb+") as file_object:
            shutil.copyfileobj(upload_file.file, file_object)
        
        segment_processor.parse_image((box_model, seg_model), new_image_file, render, dataset_labels=dataset_labels)
        if render:
            response.headers["render"] = Path(render_file).name
        # return FileResponse(segment_file,headers=response.headers, media_type='image/png', background=BackgroundTask(os.remove,segment_file))
        return FileResponse(segment_file,headers=response.headers, media_type='image/png')
    except Exception as err:
        logging.error(err)
    finally:
        # background_tasks.add_task(delete_file, render_file)
        background_tasks.add_task(delete_file, segment_file)
        Path(new_image_file).unlink(True)
        
        # Path(segment_file).unlink(True)

    
    

@app.get("/render/{render_file}")
def draw_render_file(render_file: str):
    new_image_file = os.path.join(TEMP_DIR, render_file)
    logging.info(f'Render read: {render_file}')
    return FileResponse(new_image_file, media_type='image/jpeg')


def delete_file(file:str):
    file = Path(file)
    if file.exists():
        file.unlink(True)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    uvicorn.run('webapi:app', host="0.0.0.0", port=8011, reload=True, workers =1)


