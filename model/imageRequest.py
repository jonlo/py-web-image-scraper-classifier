from pydantic import BaseModel

class ImageRequest (BaseModel):
	url: str
	image:str