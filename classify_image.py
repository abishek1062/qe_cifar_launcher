import argparse
import torch
from torch.nn import Softmax
import numpy as np
from preprocessImage import *
from model import *
import pprint


def recognizeImage(uploadfile,localfile):
    if (localfile and uploadfile) or (not(localfile) and not(uploadfile)):
        return {'error' : 'either upload an image OR give path to local DFS file', 'message' : 'failure!'}

    elif uploadfile and not(localfile):
        imagefile = uploadfile
    
    elif localfile and not(uploadfile):
        imagefile = localfile


    gpu_available = torch.cuda.is_available()

    image = get_image(imagefile)

    if image.shape[1:] != (3,32,32):
        return {'error' : "only RGB image 32x32 accepted", 'message' : "failure!" }

    model = get_model()

    output_tensor = model(image.cuda())
    output_tensor = Softmax(dim=1)(output_tensor)
    prob_pred_tensor, pred_tensor = torch.max(output_tensor, 1)

    output = np.squeeze(output_tensor.detach().numpy()) if not gpu_available else np.squeeze(output_tensor.cpu().detach().numpy())

    prob_pred = np.squeeze(prob_pred_tensor.detach().numpy()) if not gpu_available else np.squeeze(prob_pred_tensor.cpu().detach().numpy())    
    pred = np.squeeze(pred_tensor.detach().numpy()) if not gpu_available else np.squeeze(pred_tensor.cpu().detach().numpy())

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    pred_dict = {'predicted_class' : str(classes[pred]),
                 'prob_predicted_class' : str(prob_pred)}

    for i,prob in enumerate(output):
        pred_dict[classes[i]] = str(prob)


    return {'prediction' : pred_dict, 'message' : "success!" }


#driver code
pp = pprint.PrettyPrinter(indent=4)
parser = argparse.ArgumentParser(description="PyTorch Model trained on cifar-10 to classify images")
parser.add_argument("--file",type=str, required=False, help="an RGB image file of size 32x32 in either of the classes ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']",default=None)
parser.add_argument("--localfile",type=str,required=False,help="path to local DFS file",default=None)

args = parser.parse_args()
pp.pprint(recognizeImage(args.file,args.localfile))
