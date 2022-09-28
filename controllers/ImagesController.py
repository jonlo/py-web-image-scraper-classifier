import scrapper.imageScrapper as ImageScrapper
import files.ImageFiles as ImageFiles
class ImagesController:
	
	def scrap(url):
		print(url)
		return ImageScrapper.scrap(url)

	def get_image_path(url,image_name):
		return ImageFiles.get_image_path(url,image_name)