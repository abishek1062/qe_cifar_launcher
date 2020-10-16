import torch
import cv2
# import base64
# import io
# from PIL import Image
import numpy as np

def get_image(imagefile):
	# reading image
	image = cv2.cvtColor(cv2.imread(imagefile),cv2.COLOR_BGR2RGB)

	image = ((image/255) - 0.5)/0.5
	image = np.transpose(image,(2,1,0))
	image = np.expand_dims(image,axis=0)
	image = torch.from_numpy(image)
	image = image.to(torch.float32)

	return image