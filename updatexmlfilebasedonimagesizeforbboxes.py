import xml.etree.ElementTree as ET
import os

BASE_PATH = 'dataset'

Train_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations.xml"])

tree = ET.parse(Train_ANNOTS_PATH)
root = tree.getroot()
i = 0
# script to make changes in xml file when width of image is different than other images
for image in root.iter('image'):
    # print("HHHH")
    # script for chnagening bboxes dimesions according to cropped image
    if (image.attrib["width"]== "840"):
        image.attrib["width"] = "651"
        image.attrib["height"] = "510"
        for boximage in image.iter('box'):

            boximage.attrib["xtl"] = '%.3f'%(float(boximage.attrib["xtl"])-109)
            boximage.attrib["ytl"] = '%.3f'%(float(boximage.attrib["ytl"])-50)
            boximage.attrib["xbr"] = '%.3f'%(float(boximage.attrib["xbr"])-109)
            boximage.attrib["ybr"] = '%.3f'%(float(boximage.attrib["ybr"])-50)
            print(boximage.attrib, i)

        #         print(boximage.attrib, i)
    # if (image.attrib["width"] == "1920"):
    #     image.attrib["width"] = "840"
    #     image.attrib["height"] = "630"
    #     i =i+1
    #     for boximage in image.iter('box'):
    #
    #         boximage.attrib["xtl"] = '%.3f'%(float(boximage.attrib["xtl"])*(840/1920))
    #         boximage.attrib["ytl"] = '%.3f'%(float(boximage.attrib["ytl"])*(630/899))
    #         boximage.attrib["xbr"] = '%.3f'%(float(boximage.attrib["xbr"])*(840/1920))
    #         boximage.attrib["ybr"] = '%.3f'%(float(boximage.attrib["ybr"])*(630/899))
    #         print(boximage.attrib, i)
    # else:
    #     image.attrib["width"] = "840"
    #     image.attrib["height"] = "630"
    #     i = i + 1
    #     for boximage in image.iter('box'):
    #         boximage.attrib["xtl"] = '%.3f' % float(boximage.attrib["xtl"])
    #         boximage.attrib["ytl"] = '%.3f' % float(boximage.attrib["ytl"])
    #         boximage.attrib["xbr"] = '%.3f' % float(boximage.attrib["xbr"])
    #         boximage.attrib["ybr"] = '%.3f' % float(boximage.attrib["ybr"])
    #         print(boximage.attrib, i)


tree = ET.ElementTree(root)
tree.write('dataset/annotations.xml', xml_declaration=True, encoding='utf-8')