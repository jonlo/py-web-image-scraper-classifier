from fastapi import FastAPI
from routers import images

app = FastAPI()

app.include_router(images.router)

@app.get("/")
async def root():
    return {"message": "Take a look at the api calls at /docs"}
