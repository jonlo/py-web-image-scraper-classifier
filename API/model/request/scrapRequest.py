from pydantic import BaseModel

class ScrapRequest (BaseModel):
	url: str