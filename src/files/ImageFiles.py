
import os 
parent_dir_path = os.path.dirname(os.path.realpath(__file__))

def get_image_path(url,image):
	folder = get_path_for_url(url)
	dirname =os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..', 'downloads'))
	print(dirname + "/"+folder)
	return dirname + "/"+folder +"/"+image+".jpg"

def get_path_for_url(url):	
	folder_name = url.replace("https://", "").replace("http://", "").replace("www.", "").replace("/", "")
	return folder_name