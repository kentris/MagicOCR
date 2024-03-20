import os
import argparse
from magic_ocr import MagicOCR


def main(filepath:str) -> None:
    """
    A sample use-case of the MagicOCR class to extract the card names of
    Magic the Gathering cards in an image file. 

    Parameters
        ----------
        filepath : string
            The path to either a valid image file or a directory of image
            files.
    """
    magicocr = MagicOCR()
    magicocr.select_sets() # Specify which card sets to consider; default to all

    final_results = []
    # Process all image files in the specified directory
    if os.path.isdir(filepath):
        valid_ext = ['.png', '.jpg', '.gif', '.bmp']
        files_to_process = [filename for filename in os.listdir(filepath) 
                            if os.path.splitext(filename)[1].lower() in valid_ext]
        for ftp in files_to_process:
            result = magicocr.process_image(os.path.join(filepath, ftp))
            final_results += result

    # Process the specified image file
    else:
        result = magicocr.process_image(filepath)
        final_results += result

    # Print out results to command line
    for fr in final_results:
        print(fr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Filepath to the image or directory of images to be processed.")
    args = parser.parse_args()
    main(args.f)
