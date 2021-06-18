import argparse
import cv2



def inspect_image(img_path, type):
    if type == "image":
        img = cv2.imread(img_path)
        if img is not None:
            height, widhth, channel = img.shape
            print("Height x Width x Channels of the image is " + str(height) + "x" + str(widhth) + "x" + str(channel))
        else:
            print("Unable to open the image is the deired path")
    elif type == "label":
        img = cv2.imread(img_path,0)
        if img is not None:
            height, width = img.shape
            print("Height x Width of the label is "+ str(height) + "x" + str(width))
        else:
            print("Unable to open the label in specified path")
    else:
        print("The type could only be image or label")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="path to the image to be inspected")
    parser.add_argument("type", type=str, help="label or image")
    args = parser.parse_args()
    inspect_image(args.img_path, args.type)
