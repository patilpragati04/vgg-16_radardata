import xml.etree.ElementTree as ET
import os

Train_IMAGES_PATH = 'image2.jpg'
train_data = []
classes = ['Human','No Human']
trainFilenames = []
train_imagePaths = []
labels = []
trainbboxes = []
# def replacespcetounderscore():
BASE_PATH = "dataset"
# Train_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "JOB1/Annotations"])
# xmls = os.listdir(Train_ANNOTS_PATH)
# for xml in xmls:
#     # name = xml[:-4]
#     # print(name)
#     # a = name.replace(" ","_")
#     os.rename(os.path.join(Train_ANNOTS_PATH, xml), os.path.join(Train_ANNOTS_PATH, xml.replace(' ', '_')))
#
#     # print(a)
#
def convert_annotation(image_id,list_file):
    in_file = open('dataset/JOB2/Annotations/%s.xml' % image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()
    object_detected = False
    for child in root:
        # print(child.tag, child.attrib)
        train_imagePath= Train_IMAGES_PATH
        Filenames = root.find('filename').text
        if child.tag == 'object':
            cls = child.find('name').text
            labels.append(cls)
            difficult = child.find('difficult').text
            object_detected = True
            cls_id = classes.index(cls)
            xmlbox = child.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymax').text))
            trainbboxes.append(b)
            list_file.write(" " + ",".join([str(a) for a in b]) )#+ ','+str(cls_id)

    if object_detected == False:
        cls = "No Human"
        labels.append(cls)
        cls_id = classes.index(cls)
        difficult = 1
        b = (0, 0, 0, 0)
        trainbboxes.append(b)
        list_file.write(" " + ",".join([str(a) for a in b]) ) #+ ','+str(cls_id)
    # list_file.write('\n')
    return list_file


print(labels)
# print(Filenames)
print(trainbboxes)
image_extension = 'jpg'
BASE_PATH = "dataset"
Train_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "JOB2/Annotations"])
xmls = os.listdir(Train_ANNOTS_PATH)
train = open(r'info/IOU.txt', 'w')
for xml in xmls:
    # print(xml)
    name = xml[:-4]
    a = name.replace(" ","_")
    train.write('dataset/JOB2/JPEGImages/%s.%s' % (name, image_extension))
    train = convert_annotation(name,train)
    train.write('\n')


