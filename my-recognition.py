#Object recognition model
#!/usr/bin/python3

#importing jetson modules
import jetson_inference
import jetson_utils

#importing command line parser
import argparse

#parses image file name and selects network
#defines expected command line arguments, parses through them
parser = argparse.ArgumentParser() 
#file name that is passed into command line
parser.add_argument("filename", type = str, help = "filename of the image to process") 
#network that will be used
parser.add_argument("--network", type = str, default = "googlenet", help = "model to use, googlenet is default") 
#converts arguments from command line into object opt
opt = parser.parse_args()


#loads image
img = jetson_utils.loadImage(opt.filename)

#load recognition network from command line
net = jetson_inference.imageNet(opt.network)

#class_idx - index of predicted class
#confidence - accuracy of prediction
class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print("image is recognized as " + str(class_desc) + "(class #" + str(class_idx) + ") with confidence: " + str(confidence*100) + "%")