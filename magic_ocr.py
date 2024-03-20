from magic_rcnn import MagicRCNN
from card_matcher import CardMatcher
import easyocr
from PIL import Image
import cv2
import os


class MagicOCR():
    """
    The driver class for performing OCR on Magic the Gathering cards. This
    creates an instance of our MagicRCNN to predict on images, an instance
    of easyocr to perform the actual text extraction, as well as an
    instance of the CardMatcher to best map the extracted text from the 
    cards to valid card names. 
    """
    def __init__(self):
        self.magicrcnn = MagicRCNN()
        self.cardmatcher = CardMatcher()
        self.reader = easyocr.Reader(['en'])
        self.images = []
        self.text = []
        self.valid_ext = ['.png', '.jpg', '.gif', '.bmp']

    def select_sets(self, card_subsets:list=None) -> None:
        """
        Select the card sets that we will consider when mapping the extracted 
        OCR card text. Defaults to looking at all card sets in the provided file. 

        Note: Just calling the CardMatcher's `select_sets()` method. 

        Parameters
        ----------
        card_subsets : List(string)
            Should contain only valid MTG card set abbreviations (e.g. "MKM", "LCI")
        """
        self.cardmatcher.select_sets(card_subsets)

    def process_image(self, img_path:str) -> list[str]:
        """
        Load in an image file, have the RCNN predict the location of Bboxes in the
        image, perform OCR on those Bbox sub-indexed images, and then map the 
        extracted text to the best scoring match in the specific card sets. These
        final mapped results are then returned to the user. 

        Parameters
        ----------
        img_path : string
            The path to a valid image file that should contain MTG cards.

        Returns
        -------
        card_names : List(string)
            All identified card names of the cards found in the submitted image.
        """
        file_ext = os.path.splitext(img_path)[1].lower()
        if not file_ext in self.valid_ext:
            raise TypeError("Input file must be a valid image file.")
        
        # Load in image, resulting object is a numpy ndarray
        img = cv2.imread(os.path.join(img_path))

        # Feed the loaded image into the model and get predicted bounding boxes
        bboxes = self.magicrcnn.predict(img)

        card_names = []
        # For each Bbox
        for bbox in bboxes:
            # 1. Sub-index image and feed into OCR
            ocr_result = self.reader.readtext(img[bbox.y1:bbox.y2, bbox.x1:bbox.x2, :])
            if ocr_result: # Make sure we handle empty OCR results
                # easyocr may break up text if there is a large enough angle of incline on the text
                # easyocr results are formatted: ([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], <text_returned>, <score>)
                # We only care about x1 for sorting - these values should be on an approximate horizontal line
                sorted_result = sorted(ocr_result, key=lambda x: x[0][0][0])
                # And then we Join List to make all results a single string name for the card
                card_name = " ".join(sr[1] for sr in sorted_result)
                # 2. Match OCR text to card
                matched_card_name = self.cardmatcher.match(card_name)
                card_names.append(matched_card_name)

        return card_names

        