import scrapper.imageScrapper as ImageScrapper

class ImagesController:
	
	def images(url):
		print(url)
		return ImageScrapper.scrap(url)
		