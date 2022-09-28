from fastapi import FastAPI
from routers import images
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(images.router)


try:
    os.mkdir("downloads")
except:
    pass

@app.get("/")
async def root():
    return {"message": "Take a look at the api calls at /docs"}
