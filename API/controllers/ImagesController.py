import src.scrapper.imageScrapper as ImageScrapper
import src.files.ImageFiles as ImageFiles
import src.imageclassificator.imageclassificator as imageClassificator
from API.model.response.classifyResponse import ClassifyData

class ImagesController:
	
	def scrap(url):
		print(url)
		return ImageScrapper.scrap(url)

	def get_image_path(url,image_name):
		return ImageFiles.get_image_path(url,image_name)

	def classify(url):
		scrap_data = ImageScrapper.scrap(url)
		image_classes = []
		for image in scrap_data.images:
			image_path = ImageFiles.get_image_path(url,image)
			image_class = imageClassificator.classify(image_path)
			if image_class != None:
				image_classes.append({'id':image, 'class': image_class})
			print(image_class)
		data = ClassifyData(image_classes)
		return data
		