
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os


image_types = (".mp4", ".avi")
root_path = sys.argv[1]
folder_name = sys.argv[2]
video_dir = os.path.join(root_path, folder_name)

def list_videos(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed    

            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


videoPaths = list(list_videos(video_dir))
print("number of videos found", len(videoPaths))
all_frames = []
for path in videoPaths[:1]:
    cap = cv2.VideoCapture(path)
    w = 128
    h = 128
    fc = 0
    ret = True

    while True:
        ret, frame = cap.read()
        if ret == True:
            resized_frame = cv2.resize(frame, (128, 128))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
            all_frames.append(np.array(resized_frame,dtype=np.float64) / 256.0)
        else:
            break
    cap.release()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


all_frames = np.array(all_frames)
print(all_frames.shape)
sz = all_frames.shape[0] - 7
sequences = np.zeros((sz, 7, 128, 128, 3))

for i in range(0, sz):
  clip = np.zeros((7, 128, 128, 3))
  for j in range(0, 7):
    clip[j] = all_frames[i + j, :, :, :]
  sequences[i] = clip

print("sequence",sequences.shape)

padding=np.zeros((7,7,128,128,3))

for i in range(0,7):
    im=np.zeros((7,128,128,3))
    for j in range(0,7):
        im[j]=all_frames[sequences.shape[0]+i,:,:,:]
    padding[i]=im

test_data=np.concatenate((sequences, padding), axis=0)


print("test data shape",test_data.shape)

g=load_model('7ave.h5')

gan_x=g.predict(test_data)

def plot_save():
    count=0
    for i in range(gan_x.shape[0]):
        for j in range(7):
            test_image = gan_x[i, j, :, :, :]
            test_image = np.reshape(test_image, (128, 128,3))
            minv=np.min(test_image)
            maxv=np.max(test_image)
            new_image=(test_image-minv)/(maxv-minv)
            plt.imshow(test_image)
            plt.show()
            filename = image_dir + "Reconstructed_image" + str(count) + ".png"
            if new_image.min()>=0.0 and new_image.max() <=1.0:
                plt.imsave(filename, new_image)
                count+=1
            else:
                print("new image internal  {}".format(new_image))

plot_save()

sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(test_data[i],gan_x[i])) for i in range(0,test_data.shape[0])])

sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
sr = 1.0 - sa
threshold=0.7
score=sr.tolist()
result=[]
for x,y in enumerate(score):
    if y>threshold:
        result.append("anomaly")
    else:
        result.append("not anomaly")

da={'result':result,'score':score}
df=pd.DataFrame(da)
df.to_csv("path to save anomaly score" + folder_name + ".csv")

print("reconstruction cost saved numpy file")
plt.plot(sr)
plt.title(SINGLE_TEST_PATH)
plt.ylabel('regularity score Sr(t)')
plt.xlabel('frame t')
plt.show()
