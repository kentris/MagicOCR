import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy
from collections import namedtuple
import os

Bbox = namedtuple('BoundingBox', ['x1', 'y1', 'x2', 'y2'])
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)


def get_iou(bb1:Bbox, bb2:Bbox) -> float:
    """
    Reference: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : Bbox(namedtuple)
        variables: 'x1', 'x2', 'y1', 'y2'
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : Bbox(namedtuple)
        variables: 'x1', 'x2', 'y1', 'y2'
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1.x1 < bb1.x2
    assert bb1.y1 < bb1.y2
    assert bb2.x1 < bb2.x2
    assert bb2.y1 < bb2.y2

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1.x1, bb2.x1)
    y_top = max(bb1.y1, bb2.y1)
    x_right = min(bb1.x2, bb2.x2)
    y_bottom = min(bb1.y2, bb2.y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1.x2 - bb1.x1) * (bb1.y2 - bb1.y1)
    bb2_area = (bb2.x2 - bb2.x1) * (bb2.y2 - bb2.y1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

class MagicRCNN():
    """
    A Recurrent Convolutional Neural Network that has been trained to identify
    the card name section of Magic the Gathering Cards. It was fine-tuned from
    the FasterRCNN Resnet50 default weights available in torchvision, and the 
    head has been replaced to only identify MTG card names (i.e. 2 classes,
    background and the card names). )
    """
    def __init__(self, path:str=os.path.join(__location__, "magic_rcnn.pt")):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        ## Load our model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            )
        
        # Make our edits to the head of the original model
        num_classes = 2 
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(path))
        model.eval() # Put into evaluation mode
        self.model = model.to(self.device)

        # The transform to be used for converting images to tensors
        self.transform = T.ToTensor()

    def predict(self, img:numpy.ndarray) -> list[Bbox]:
        """
        Predict the 

        Parameters
        ----------
        img : numpy.ndarray
            The numpy representation of the loaded image - should contain
            at least one MTG card. 

        Returns
        -------
        final_bboxes : List[Bbox]
            A list containing the bounding boxes of all identified 
            card names in the submitted image. 
        """
        # Convert numpy array image to tensor representation
        img = self.transform(img)

        with torch.no_grad():
            pred = self.model([img.to(self.device)])

        bboxes, _, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]

        # Track the filtered boxes we're interested in
        # 1. Has a matching score of greater than 80%
        # 2. Has a less than 35% intersection with any previously identified Bbox 
        #    this will help ensure we grab SEPARATE cards and not simply overlapping bboxes of the same card
        score_threshold = 0.10
        iou_threshold = 0.35
        final_bboxes = []
        for bbox, score in zip(bboxes, scores):
            b = Bbox(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])) # Bounding boxes come back as Tensor(x1, y1, x2, y2)
            max_iou = 0 if len(final_bboxes)==0 else max([get_iou(fb, b) for fb in final_bboxes])
            if score > score_threshold and max_iou < iou_threshold:
                final_bboxes.append(b)

        final_bboxes = sorted(final_bboxes, key=lambda x: int(x.x1/500)*1000 + int(x.y1/100))
        return final_bboxes
