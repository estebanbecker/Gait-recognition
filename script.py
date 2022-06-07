from cgi import test
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

path_dataset="./GaitDataset"
train_walk = ['nm-03', 'bg-01', 'nm-06', 'nm-04', 'cl-01']
point_of_view="090"


def convert(path, target_size):
    img_list=[]

    for file in sorted(os.listdir(path)):
        
        img=cv2.imread(path+"/"+file,cv2.IMREAD_GRAYSCALE)
        ret,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

        x,y,w,h = cv2.boundingRect(img)

        if x!=0 and y != 0 and w!= 0 and h!=0:

            img = cv2.resize(img[y:y+h,x:x+w],target_size)


            img_list.append(img)

    if img_list == None:
        return None
    GEI=np.mean(img_list,axis = 0)

    return GEI.flatten()

train_data = []
train_labels = []

test_data = []
test_labels = []

for folders in os.listdir(path_dataset):

    for walking in os.listdir(path_dataset+"/"+folders):

        path=path_dataset+"/"+folders+"/"+walking+"/"+point_of_view

        GEI = convert(path,(70,210))


        if walking in train_walk:

            train_data.append(GEI)
            train_labels.append(folders)
        
        else:

            test_data.append(GEI)
            test_labels.append(folders)

classifier= RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1, max_depth=100, max_features=100)
classifier.fit(train_data,train_labels)

good_count=0



prediction = classifier.predict(test_data)

for i in range(len(prediction)):
    if prediction[i] == test_labels[i]:

        good_count += 1

    
print(good_count/len(test_labels))

#With the training walk: ['nm-03', 'bg-01', 'nm-06', 'nm-04', 'cl-01']
#and the point of view = 90 degrees
#We obtain a precision of 0.95 

#Now, using the same train set but with a point of view of 36 degrees we obtain a precision of 0.92

#Finally with the type of walk: ['nm-04', 'nm-05', 'nm-02', 'cl-02']
#and the point of view 90 degrees
#We obtain a precision of 0.77

#We can see that it is important to have the same type of walk in the train and test dataset to have good results