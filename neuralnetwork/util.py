# This script is for converting the unicode byte files
# for the MNIST datasets available at 
# http://yann.lecun.com/exdb/mnist/ into csv files
# for testing and training the Neural network

# import the csv reader/writer
import csv
import requests
import gzip

# function to combine and convert 
# the ubyte files for the images 
# and the labels into one
def convert(img_file, label_file, output_file, lines):
    
    # Open each of the files
    img = open(img_file, "rb")
    lbl = open(label_file, "rb")
    
    # cycle down the first few btis of the files
    img.read(16)
    lbl.read(8)
    images = []
    
    for image in range(lines):
        image = [ord(lbl.read(1))] # read the label
        for pix in range(28*28): # num of pixels
            image.append(ord(img.read(1)))
        images.append(image)
        
    # write the images to the output csv file
    with open(output_file, "w") as csvfile:
        filewriter = csv.writer(csvfile)
        for image in images:
            # write out the csv file from the list of images
            filewriter.writerow(image)
            
    # Close files and flush buffers from memory        
    img.close()
    lbl.close()

def retrieve(url, filenames, ext):

	# Download the list of files from the base url
	for file in filenames:
		response = requests.get(url+file+ext, stream=True)
		with open(file+ext, "wb") as handle:
			print("Downloading", file+ext)
			for data in response.iter_content():
				handle.write(data)


def decompress(filenames, ext):
	
	# used to decompress the list of archived files	
	for file in filenames:	
		with gzip.open(file+ext, "rb") as container, open(file, "wb") as new:
			print("Extracting", file)
			new.write(container.read())
