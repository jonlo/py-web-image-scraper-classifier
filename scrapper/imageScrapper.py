from bs4 import BeautifulSoup
import requests
import os
from .scrapData import ScrapData

def scrap(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
    r= requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    images = soup.findAll('img')
    scrapdata = folder_create(images,url)
    return scrapdata

def folder_create(images,url):
    try:
        os.mkdir("downloads")
    except:
        pass
    try:
        folder_name = url.replace("https://", "").replace("http://", "").replace("www.", "").replace("/", "")
        path = "downloads/"+folder_name
        os.mkdir(path)
    except:
        pass
    images = download_images(images, path)
    scrapdata = ScrapData(folder_name, images)
    return scrapdata
 
def download_images(images, path):
    count = 0
    print(f"Total {len(images)} Image Found!")
    image_names = []
    if len(images) != 0:
        for i, image in enumerate(images):
            image_link = get_image_source(image)
            try:
                r = requests.get(image_link).content
                try:
                    r = str(r, 'utf-8')
                except UnicodeDecodeError:
                    image_name = clean_image_name(image_link)
                    image_names.append(image_name)
                    with open(f"{path}/images{image_name}.jpg", "wb+") as f:
                        f.write(r)
                    count += 1
            except:
                pass
        if count == len(images):
            print("All Images Downloaded!")
        else:
            print(f"Total {count} Images Downloaded Out of {len(images)}")
    return image_names

def get_image_source(image):
    try:
        image_link = image["data-srcset"]
    except:
        try:
            image_link = image["data-src"]
        except:
            try:
                image_link = image["data-fallback-src"]
            except:
                try:
                    image_link = image["src"]
                except:
                    try:
                        image_link = image["data-sf-src"]
                    except:
                        pass
    return image_link
def clean_image_name(image):
	return image.replace("https://", "").replace("http://", "").replace("www.", "").replace("/", "").replace(".jpg", "").replace(".png", "").replace(".jpeg", "")