from flask import Flask, render_template, request
import glob
import os
import cv2
import time
import torch
from model import myIFCNN
from PIL import Image
from utils.myTransforms import denorm, norms, detransformcv2
import numpy as np
app = Flask(__name__)

@app.route('/')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
   if request.method == 'POST':
      f = request.files['file']
      f.save('static\\'+secure_filename(f.filename))
      return 'file uploaded successfully'
@app.route('/extra',methods = ['GET','POST'])
def operation():
    if request.method == 'POST':
      f = request.files['file1']
      f2 = request.files['file2']
      f.save('static\\'+'img1.tif')
      f2.save('static\\'+'img2.tif')
    else:
        return "error"
    dwtimage = glob.glob('static/*.tif')

    from torchvision import transforms
    from torch.autograd import Variable

    fuse_scheme = 2
    if fuse_scheme == 0:
        model_name = 'IFCNN-MAX'
    elif fuse_scheme == 1:
        model_name = 'IFCNN-SUM'
    elif fuse_scheme == 2:
        model_name = 'IFCNN-MEAN'
    else:
        model_name = 'IFCNN-MAX'

    # load pretrained model
    device = torch.device('cpu')
    model = myIFCNN(fuse_scheme=fuse_scheme)
    model.load_state_dict(torch.load('snapshots/' + model_name + '.pth', map_location=device))
    model.eval()
    model = model.to(device)

 
    path1 = os.path.join(dwtimage[0])
    path2 = os.path.join(dwtimage[1])
    is_save = True
    is_gray = True
    mean=[0, 0, 0]         # normalization parameters
    std=[1, 1, 1]
    from utils.myDatasets import ImagePair
    pair_loader = ImagePair(impath1=path1, impath2=path2, 
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)
                                      ]))
    img1, img2 = pair_loader.get_pair()
    img1.unsqueeze_(0)
    img2.unsqueeze_(0)

    # perform image fusion
    with torch.no_grad():
        img1 = img1.to(device)
        img2 = img2.to(device)
        res = model(Variable(img1), Variable(img2))
        res = denorm(mean, std, res[0]).clamp(0, 1) * 255
        res_img = res.cpu().data.numpy().astype('uint8')
        img = res_img.transpose([1,2,0])

    # save fused images
    if is_save:
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = Image.fromarray(img)
            img.save('static/result/'+'res.png', format='PNG', compress_level=0)
        else:
            img = Image.fromarray(img)
            img.save('static/result/'+'res.png', format='PNG', compress_level=0)
    return render_template('result.html')
if __name__ == '__main__':
   app.run(debug = True)

   