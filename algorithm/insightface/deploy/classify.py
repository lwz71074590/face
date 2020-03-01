import face_model
import argparse
import cv2
import sys
import numpy as np
import time
import sklearn
import os
import math
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import pickle
import facenet
from sklearn.svm import SVC

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model, 0', help='path to load model.')
parser.add_argument('--ga-model', default='../models/model-r34-amf/model, 0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def load_data(image_paths, image_size):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, 3, image_size, image_size))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (image_size, image_size))
        img = np.transpose(img, (2,0,1))
        images[i,:,:,:] = img
    return images


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

def compare_image():
    model = face_model.FaceModel(args)
    img1 = cv2.imread('./test_image/1.png')
    img2 = cv2.imread('./test_image/85.jpg')

    time1 = time.time()
    img1, _ = model.get_input(img1)
    img2, _ = model.get_input(img2)
    time2 = time.time()
    #f1 = model.get_feature(img)
    #print(f1[0:10])
    print(img1.shape)
    f1 = model.get_feature(img1).flatten()
    f2 = model.get_feature(img2).flatten()
    sim = sklearn.metrics.pairwise.cosine_similarity([f1, f2])
    print(sim)
    time3 = time.time()



def train_classify(face_dir, mode, batch_size, min_nrof_images_per_class, nrof_train_images_per_class, classifier_filename, train_split_rate=1):
    #  加载模型
    model = face_model.FaceModel(args)
    # 获取dataset
    if train_split_rate < 1:
        dataset_tmp = get_dataset(face_dir)
        train_dataset, test_dataset = split_dataset(dataset_tmp, min_nrof_images_per_class, nrof_train_images_per_class)
        if mode == 'TRAIN':
            dataset = train_dataset
        if mode == 'TEST':
            dataset = test_dataset
    else:
        dataset = get_dataset(face_dir)
     # Check that there are at least one training image per class
    for cls in dataset:
        assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')  

    paths, labels = get_image_paths_and_labels(dataset)
            
    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))
    
    print('Calculating features for images')
    nrof_images = len(paths)
    nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
    embedding_size = 512
    image_size = 112
    
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches_per_epoch):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = load_data(paths_batch, image_size)
        # images = facenet.load_images(images, False, False, image_size)
        # images = np.transpose(images, (0, 3, 1, 2))
        emb_array[start_index:end_index, :] = np.reshape(model.get_feature(images), (-1, 512))
    classifier_filename_exp = os.path.expanduser(classifier_filename)
    if mode == 'TRAIN':
        print('Training classifier')
        # model = neighbors.KNeighborsClassifier()
        # model = tree.DecisionTreeClassifier()
        # param = {'max_depth':8, 'eta':0.2, 'n_estimators':1000, 'objective':'mulit:softmax' }
        # model = XGBClassifier(max_depth=8, n_estimators = 2000, objective='mulit:softmax')
        model = SVC(kernel='rbf', probability=True)
        # model = RandomForestClassifier(n_estimators=500, max_depth=None)
        model.fit(emb_array, labels)
        
        # Create a list of class names
        class_names = [ cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
    if mode == 'TEST':
        # Classify images
        print('Testing classifier')
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % classifier_filename_exp)
        print(emb_array)
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        
        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
            
        accuracy = np.mean(np.equal(best_class_indices, labels))
        print('Accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    # train_classify('../../../demo/sdkj1/train', 'TEST', 48, 2, 95, 'random_forest_sdkj.pkl', train_split_rate=1)
    # train_classify('/media/heyulin/5D08C9582606C5A3/demo/sdkj1/test', 'TEST', 48, 2, 95, 'svc_sdkj.pkl', train_split_rate=1)
    # compare_image()