from ast import For


from dataclasses import dataclass

@dataclass
class ScrapData:
	folder:str
	images:list
	def __init__(self, folder, images): 
		self.folder = folder 
		self.images = images