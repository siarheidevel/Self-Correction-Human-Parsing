import os
import torch, cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))
CUDA_DEVICE= 'cuda:0'
torch.cuda.set_device(CUDA_DEVICE)
CTX = torch.device(CUDA_DEVICE) if torch.cuda.is_available() else torch.device('cpu')
# from lib.utils.analyser import Analyser
import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
from utils.transforms import get_affine_transform
from utils.transforms import transform_logits

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}
num_classes = dataset_settings['lip']['num_classes']
input_size = dataset_settings['lip']['input_size']

reduced_dataset = { 
    'labels': {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8, 'skin':9, 'glove':10 },

    'lip2reduced': {'Background': 'background', 'Hat': 'hat', 'Hair': 'hair', 'Sunglasses': 'face', 
                    'Upper-clothes': 'upper-clothes',
                    'Dress': 'upper-clothes', 'Coat': 'upper-clothes',
                    'Socks': 'shoes', 'Pants': 'pants', 'Scarf': 'upper-clothes', 'Skirt': 'pants', 
                    'Face': 'face', 'Left-arm': 'arm', 'Right-arm': 'arm',
                    'Left-leg': 'leg', 'Right-leg': 'leg', 'Left-shoe': 'shoes', 'Right-shoe': 'shoes', 
                    'Gloves':'glove', 'Torsoskin':'skin'}
    
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def draw_legend(colors,labels, imsize=(200,600)):
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as mpatches
    # labels = [k for k,v in reduced_dataset['labels'].items()]
    # colors = np.array(get_palette(len(reduced_dataset['labels']))).reshape(len(reduced_dataset['labels']), 3)

    # create a patch (proxy artist) for every color 
    # patches = [ mpatches.Patch(color=colors[i]/255, label=labels[i]) for i in range(len(labels)) ]
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    # cv2.

    # initialize the legend visualization
    legend = np.zeros((*imsize, 3), dtype="uint8")
    # loop over the class names + colors
    per_row = int(imsize[0]/len(labels))
    for i in range(len(labels)):
        color = colors[i]
        label = labels[i]
        cv2.putText(legend, label, (5, (i * per_row) + 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(legend, (100, (i * per_row)), (imsize[1], (i * per_row) + 25),
            list(reversed(color.tolist())), -1)

    return legend


def get_person_detection_boxes(model, img, threshold=0.5, expand_box_percent = 0.05):
    '''
    returns bboxes (w0,h0),(w1,h1)
    '''
    pred = model(img)
    height, width = img[0].shape[1:]
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            # person_boxes.append(box)
            expand_w, expand_h = int(expand_box_percent * (box[1][0] - box[0][0])), int(expand_box_percent * (box[1][1] - box[0][1]))
            person_boxes.append([(max(0,box[0][0]-expand_w),max(0,box[0][1]-expand_h)), 
                (min(width,box[1][0]+expand_w),min(height,box[1][1]+expand_h))])

    return person_boxes


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def load_model():
    box_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    seg_model = networks.init_model(
        'resnet101', num_classes=num_classes, pretrained=None)
    state_dict = torch.load(
        'saved_models/exp-schp-201908261155-lip.pth')['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    seg_model.load_state_dict(new_state_dict)
    # if 'cuda'==device and torch.cuda.is_available():
    #     net.cuda()
    seg_model = seg_model.to(CTX)
    seg_model.eval()
    return box_model, seg_model


def parse_image(model, image_file, do_render:bool = True, extension: str ='', dataset_labels='redused'):
    box_model, seg_model = model
    im = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    orig_im = im
    image_width, image_height = (im.shape[1], im.shape[0])

    input = []
    image_rgb = im[:, :, [2, 1, 0]]
    img_tensor = torch.from_numpy(image_rgb/255.).permute(2,0,1).float().to(next(box_model.parameters()).device)
    input.append(img_tensor)
    # person detection box
    pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
    if len(pred_boxes) >= 1:
        for box in pred_boxes:
            # Image.fromarray(image_rgb[int(pred_boxes[0][0][1]):int(pred_boxes[0][1][1]),int(pred_boxes[0][0][0]):int(pred_boxes[0][1][0]),...]).save("box_img.png")
            # 384, 512
            box_img_array = im[int(box[0][1]):int(box[1][1]),int(box[0][0]):int(box[1][0]),...]
            model_height, model_width = input_size#seg_model.cfg.TEST.SCALE[1],seg_model.cfg.TEST.SCALE[0]
            box_height, box_width = int(box[1][1]) - int(box[0][1]), int(box[1][0]) - int(box[0][0])
            scale = min(model_height/box_height, model_width/box_width)
            dy = (model_height - scale*box_height)/2
            dx = (model_width - scale * box_width)/2
            transMatrix = np.array([[scale,0,dx],[0,scale,dy]])
            box_model_array = cv2.warpAffine(
                cv2.bilateralFilter(box_img_array,5,15,15),
                transMatrix,
                (model_width, model_height),
                flags=cv2.INTER_AREA)
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                                    0.225, 0.224, 0.229])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                #                      0.229, 0.224, 0.225])
            ])
            input = transform(box_model_array).unsqueeze(0).to(CTX)
            with torch.no_grad():
                output = seg_model(input)
                upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
                # .permute(1, 2, 0)
                # parsing_seg = torch.max(output, dim=1)[1][0].cpu().numpy().astype(np.uint8)
                parsing_seg = np.argmax(upsample_output.data.cpu().numpy(), axis=2)
            
            dy = (int(box[1][1]) + int(box[0][1]))//2 - int(model_height /scale/2)
            dx = (int(box[1][0]) + int(box[0][0]))//2 - int(model_width/scale /2)
            transMatrix = np.array([[1/scale,0,dx],[0,1/scale,dy]])
            parsing_seg = cv2.warpAffine(
                parsing_seg,
                transMatrix,
                (image_width, image_height),
                flags=cv2.INTER_NEAREST).astype(np.uint8)
            break
    
        if dataset_labels == 'reduced':
            palette = get_palette(len(reduced_dataset['labels']))
            rgb_reduced_palette_colors = np.array(palette).reshape(-1, 3)
            original2reduced = np.array([reduced_dataset['labels'][reduced_dataset['lip2reduced'][category]]
                for i, category in enumerate(dataset_settings['lip']['dataset_labels']) ])
            redused_legend = draw_legend(rgb_reduced_palette_colors,[k for k,v in reduced_dataset['labels'].items()],
                (im.shape[0],200))
            reduced_mask = original2reduced[parsing_seg.astype(int)]
        elif dataset_labels == 'lip':
            palette = get_palette(len(dataset_settings['lip']['label']))
            rgb_reduced_palette_colors = np.array(palette).reshape(-1, 3)
            original2reduced = np.array([i for i, category in enumerate(dataset_settings['lip']['label'])])
            redused_legend = draw_legend(rgb_reduced_palette_colors,dataset_settings['lip']['label'],
                (im.shape[0],200))
            reduced_mask = parsing_seg.astype(int)        

        mask_image=Image.fromarray(reduced_mask.astype(dtype=np.uint8))
        mask_image.putpalette(palette)
        mask_image.save(f'{image_file}.seg_schp{extension}.render.png')
        
        if do_render:
            mask_rgb = rgb_reduced_palette_colors[reduced_mask]
            rendered = np.clip(orig_im * 0.5 + 0.5 *
                            mask_rgb[:, :, [2, 1, 0]], 0, 254).astype(np.uint8)
            rendered = np.concatenate((orig_im,rendered,
                mask_rgb[:, :, [2, 1, 0]],cv2.resize(redused_legend, (200,rendered.shape[0]))), axis=1)
            # np.savez_compressed(image_file+'.seg3', mask = reduced_mask.astype('uint8'))
            cv2.imwrite(f'{image_file}.seg_schp{extension}.render.jpg',cv2.resize(
                rendered.astype(float)
                ,None, fx=0.5, fy=0.5,  interpolation=cv2.INTER_AREA)
                ,[int(cv2.IMWRITE_JPEG_QUALITY), 50])




    

# if __name__=="__main__":
#     model = load_model()
#     # model = load_model_lip()
#     dir = Path('/home/deeplab/datasets/custom_fashion/demo')
#     dir=Path('/home/deeplab/datasets/deepfashion/diordataset_custom/img_highres/WOMEN/Pants/id_00000029')
#     # dir = Path('/home/deeplab/datasets/custom_fashion/data/wildberries_ru_')
#     dir = Path('/home/deeplab/datasets/custom_fashion/data/lamoda_ru_')
#     # dir = Path('/home/deeplab/datasets/custom_fashion/demo_')
#     dir = Path('/home/deeplab/datasets/deepfashion/shop_wild/img_highres')
#     # dir = Path('/home/deeplab/datasets/deepfashion/diordataset_custom/img_highres/WOMEN')
#     # dir = Path('/home/deeplab/datasets/clothing-co-parsing/photos')
#     dir = Path('/home/deeplab/datasets/custom_fashion/data/wildberries_ru_/3276')
#     # parse_dir(model, dir,extension='_lip')
#     # parse_dir(model, dir,extension='')
#     # /home/deeplab/datasets/deepfashion/diordataset_custom/img_highres/MEN/Pants
#     # parse_image(model,'/home/deeplab/datasets/custom_fashion/data/wildberries_ru_/3276/32768198/32768198-3.jpg',extension='_lip')
#     # parse_image(model, '/home/deeplab/datasets/clothing-co-parsing/photos/0001.jpg')
#     # parse_image(model, '/home/deeplab/datasets/custom_fashion/demo_/1544/15445714/15445714-1.jpg')
#     # parse_image(model, '/home/deeplab/datasets/deepfashion/diordataset_custom/img_highres/MEN/Pants/id_00000063/02_1_front.jpg')

#     # parse_image(model, '/home/deeplab/datasets/custom_fashion/data/wildberries_ru_/3276/32763551/32763551-1.jpg')
#     parse_image(model, '/home/deeplab/datasets/custom_fashion/data/wildberries_ru_/3276/32765177/32765177-2.jpg')
#     print('Finished')