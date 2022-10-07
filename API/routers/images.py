
from API.controllers.ImagesController import ImagesController
from fastapi import APIRouter,HTTPException
import API.model.request.scrapRequest as scrapRequest
import API.model.request.imageRequest as imageRequest
from API.model.response.classifyResponse import ClassifyData
from src.scrapper.scrapData import ScrapData
from fastapi.responses import FileResponse

router = APIRouter()

@router.get("/scrap", response_model=ScrapData)
async def create_item(scrap_request: scrapRequest.ScrapRequest):
    try:
        scrap_data = ImagesController.scrap(scrap_request.url)
        return scrap_data
    except:
        raise HTTPException(status_code=500, detail="Something went wrong")

@router.get("/image", response_class=FileResponse)
async def create_item(image_request: imageRequest.ImageRequest):
    try:
        image_path = ImagesController.get_image_path(image_request.url,image_request.image)
        return image_path
    except:
        raise HTTPException(status_code=500, detail="Something went wrong")
@router.get("/classify", response_model=ClassifyData)
async def create_item(classify_request: scrapRequest.ScrapRequest):
    # try:
        scrap_data = ImagesController.classify(classify_request.url)
        return scrap_data
    # except:
    #     raise HTTPException(status_code=500, detail="Something went wrong")