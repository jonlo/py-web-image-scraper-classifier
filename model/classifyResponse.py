from ast import For


from dataclasses import dataclass

@dataclass
class ClassifyData:
	images:list
	def __init__(self, images): 
		self.images = images