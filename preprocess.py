import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class ImageProcessor:
    def __init__(self, cmu_pie_dir, selfie_dir, processed_selfie_dir, img_size=(32, 32)):
        self.cmu_pie_dir = cmu_pie_dir
        self.selfie_dir = selfie_dir
        self.processed_selfie_dir = processed_selfie_dir
        self.img_size = img_size
        
        # Create processed selfie directory if it doesn't exist
        if not os.path.exists(self.processed_selfie_dir):
            os.makedirs(self.processed_selfie_dir)

    def load_cmu_pie_images(self, subjects, train_ratio=0.7):
        X_train, X_test, y_train, y_test = [], [], [], []
        for label, subject in enumerate(subjects):
            subject_dir = os.path.join(self.cmu_pie_dir, str(subject))

            images=[]
            for img in os.listdir(subject_dir):
                image = cv2.imread(os.path.join(subject_dir, img), cv2.IMREAD_GRAYSCALE)
                image = image.flatten()
                # images = [img.flatten() for img in images if img is not None]  # No resize needed
                images.append(image)
            # Split images into training and test sets for the subject
            X_sub_train, X_sub_test = train_test_split(images, train_size=train_ratio, random_state=36)
            X_train.extend(X_sub_train)
            X_test.extend(X_sub_test)
            # y_train.extend([label] * len(X_sub_train))
            y_train.extend([label] * np.shape(X_sub_train)[0])
            y_test.extend([label] * len(X_sub_test))

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def load_selfie_images(self, train_size=7):
        X_selfie, y_selfie  = [], []
        for i in range(1, 11):
            img_path = os.path.join(self.selfie_dir, f"{i}.jpg")
            save_path = os.path.join(self.processed_selfie_dir, f"{i}.jpg")
            
            if os.path.exists(save_path):  # Load processed selfie if it exists
                img_resized = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
                img_resized = img_resized.flatten()

            else:  # Process and save if not already done
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, self.img_size)
                cv2.imwrite(save_path, img_resized) 
                img_resized = img_resized.flatten()
                
            X_selfie.append(img_resized)
            # y_selfie.append(-1)  # Label for selfie images
            y_selfie.append(25)  # Label for selfie images
        
        # Split selfies into train and test
        X_selfie_train = np.array(X_selfie[:train_size])
        X_selfie_test = np.array(X_selfie[train_size:])
        y_selfie_train = np.array(y_selfie[:train_size])
        y_selfie_test = np.array(y_selfie[train_size:])
        
        return X_selfie_train, X_selfie_test, y_selfie_train, y_selfie_test

    def get_dataset(self, subjects, train_ratio=0.7):
        # Load CMU PIE images and labels
        X_train, X_test, y_train, y_test = self.load_cmu_pie_images(subjects, train_ratio)
        
        # Load selfie images and labels
        X_selfie_train, X_selfie_test, y_selfie_train, y_selfie_test = self.load_selfie_images()
        
        # Combine CMU PIE and selfies for training and testing
        X_combined_train = np.vstack((X_train, X_selfie_train))
        X_combined_test = np.vstack((X_test, X_selfie_test))
        y_combined_train = np.concatenate((y_train, y_selfie_train))
        y_combined_test = np.concatenate((y_test, y_selfie_test))
        
        return X_combined_train, X_combined_test, y_combined_train, y_combined_test
