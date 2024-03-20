# MagicOCR

**MagicOCR** is a project aimed at easily and quickly extracting card names from _Magic: The Gathering_ cards. The primary use of this application is intended to provide a tool to casual communities that enables a quick way to maintain collections without requiring they manually type in all cards in their collections. 

## Methodology
1. Recurrent Convolutional Neural Network
2. OCR on predicted Bounding Boxes
3. Mapping Extracted Text to Relevant Card Sets

The RCNN (Recurrent Convolutional Neural Network) used in this application is based on torchvision's Faster RCNN Resnet50 architecture and has default weights from the Resnet50_FPN_Weights.COCO_V1 trained model. The model is fine-tuned on a set of _Magic: The Gathering_ cards identifying the bounding boxes for the card names section of the card specifically (as opposed to the entire card as a whole). 

Once the bounding boxes for the predicted card names has been identified, the application, utilizes the _easyocr_ module to perform Optical Character Recognition (OCR) on the sub-indexed section of the image (the bounding boxes). 

The text that is extracted from these images is then mapped to the best matching card name in the specified card sets (at the time of this writing, "LCI" and "MKM" have been included in this list). "Best matching" in this instance refers to the card name that has the highest Jaro-Winkler score. The best match is then returned to the user for that bounding box/card. 

## Model
The fine-tuned model used in this application can be downloaded [here](https://drive.google.com/file/d/1uy6wjQEQHJ2mnxoOjviEJEZA7qWwh_e7/view?usp=sharing). 

## Sample Usage
The application has a simple command at this time. A sample call to the application can be seen below:
```
python main.py -f .\images\mtg_cards_01.jpg
```

The argument ```-f``` indicates the filepath of the image file or directory containing image files to be processed. All valid image files in the specified directory will be processed when the program runs. 