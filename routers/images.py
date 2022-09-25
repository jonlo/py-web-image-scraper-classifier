
from controllers.ImagesController import ImagesController
from fastapi import APIRouter,HTTPException
import model.imageRequest as imageRequest
from scrapper.scrapData import ScrapData

router = APIRouter()

@router.get("/images", response_model=ScrapData)
async def create_item(imagerequest: imageRequest.ImageRequest):
    try:
        scrap_data = ImagesController.images(imagerequest.url)
        print(scrap_data)
        return scrap_data
    except:
        raise HTTPException(status_code=500, detail="Something went wrong")