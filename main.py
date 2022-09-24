from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/images/{website}")
async def read_item(website):
    return {"website": website}