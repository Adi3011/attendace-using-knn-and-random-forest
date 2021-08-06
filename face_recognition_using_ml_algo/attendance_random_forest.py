import math
from sklearn.ensemble import RandomForestClassifier
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, verbose=False):
    """
    Trains a random forest classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(1 if class_dir=='elon' else 0)


    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(clf, f)

    return clf


def predict(X_img_path, clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained random forest classifier
    :param X_img_path: path to image to be recognized
    :param clf: (optional) a random forest classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled random forest classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh clf or model_path")

    # Load a trained random forest model (if one was passed in)
    if clf is None:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the random forest model to find the best matches for the test face
    pred = clf.predict(faces_encodings)
    
    return pred




if __name__ == "__main__":
    # STEP 1: Train the random forest classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training Random_forest classifier...")
    classifier = train("train", model_save_path="trained_random_model.clf")
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("test"):
        full_file_path = os.path.join("test", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        pred = predict(full_file_path, model_path="trained_random_model.clf")

       
        if pred==1:
            print("Elon is present")
        else:
            print("jeff is present")    

       