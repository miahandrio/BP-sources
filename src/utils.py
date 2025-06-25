import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy
from enum import Enum
from sklearn import metrics
from src.photo import Photo
from tqdm import tqdm
import random
import torch
import gc        

class Utils:
    
    @staticmethod
    def eval(func, photos, prompt, debug = False):
        with tqdm(total=len(photos), desc="Classifying Photos") as pbar:
            for photo in photos:
                result = func(photo.path, prompt)
                if (debug):
                    print(result)
                photo.classify(result)
                photo.get_name()
                pbar.update(1)
                
        def fil(photo):
            if (photo.predicted == Photo.Class.Uncategorized):
                return False
            else:
                return True
        photos = list(filter(fil, photos))
        Utils.print_metrics(photos)
        Utils.print_FP_FN(photos)
        Utils.visualize_photos(photos)

    def gen_train_dataset(func, photos, prompt, output_json = "output.json"):
        with tqdm(total=len(photos), desc="Generating training dataset") as pbar:
            dataset = []
            for photo in photos:
                result = func(photo.path, prompt)
                message = {
                    "image": photo.path,
                    "output": result
                }
                dataset.append(message)
                photo.classify(result)
                pbar.update(1)
                
        def fil(photo):
            if (photo.predicted == Photo.Class.Uncategorized):
                return False
            else:
                return True
        photos = list(filter(fil, photos))
        Utils.print_metrics(photos)
        Utils.print_FP_FN(photos)
        Utils.visualize_photos(photos)

        with open(output_json, 'w', encoding='utf-8') as f_out:
            json.dump(dataset, f_out, indent=2, ensure_ascii=False)

        print(f"Saved {len(dataset)} conversation examples to {output_json}")
        

    @staticmethod
    def print_FP_FN(photos: list):
        photos = Utils.sort_photos(photos)
        def filter_case(photo, allowed_case = ['FP', 'FN']) -> bool:
            return photo.case in allowed_case
        photos = filter(filter_case, photos)
        for photo in photos:
            print(photo.get_name())
        
    
    @staticmethod
    def list_files_recursive(paths: list, path='.'):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                Utils.list_files_recursive(paths, full_path)
            else:
                paths.append(full_path)

    @staticmethod
    def list_files(path='.') -> list:
        paths = []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if not os.path.isdir(full_path):
                paths.append(full_path)
        return paths

    @staticmethod
    def get_dataset_data(split ='test', limit = -1, base_path='./data'):
        full_path = base_path + "/" + split
        photos = []
        
        paths = Utils.list_files(full_path + "/acceptable")
        acceptable = 0;
        for path in paths:
            photos.append(Photo(path, Photo.Class.Acceptable))
            acceptable+=1
            
        paths = Utils.list_files(full_path + "/unacceptable")
        unacceptable = 0
        for path in paths:
            photos.append(Photo(path, Photo.Class.Unacceptable))
            unacceptable+=1

        print(f'A total of {len(photos)} photos were loaded as {split} dataset. Of which, {acceptable} are acceptable and {unacceptable} are unacceptable')
        random.shuffle(photos)

        if (not limit == -1):
            return photos[:limit]
        else:
            return photos

    @staticmethod
    def print_metrics(photos):
        
        actual = []
        predicted = []

        for photo in photos:
            if (photo.predicted == Photo.Class.Uncategorized):
                print(photo.get_name() + " was purged")
            actual.append(photo.actual.value)
            predicted.append(photo.predicted.value)

        labels = [Photo.Class.Acceptable.value, Photo.Class.Unacceptable.value]
    
        Utils.plot_conf_matrix(actual, predicted, labels)
        Utils.print_aprf(actual, predicted)
        Utils.plot_prc(actual, predicted)
        Utils.plot_roc_graph(actual, predicted)

    @staticmethod
    def plot_conf_matrix(actual, predicted, labels):
        confusion_matrix = metrics.confusion_matrix(actual, predicted, labels = labels)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [Photo.Class.Acceptable.value, Photo.Class.Unacceptable.value])
        cm_display.plot()
        plt.show()

    @staticmethod
    def print_aprf(actual, predicted):
        Accuracy = metrics.accuracy_score(actual, predicted)
        Precision = metrics.precision_score(actual, predicted, pos_label=Photo.Class.Acceptable.value)
        Recall = metrics.recall_score(actual, predicted, pos_label=Photo.Class.Acceptable.value)
        F1_score = metrics.f1_score(actual, predicted, pos_label=Photo.Class.Acceptable.value)
        print({"Accuracy":Accuracy,"Precision":Precision,"Recall":Recall,"F1_score":F1_score})
        
    def plot_prc(actual, predicted):
        actual_binary = [1 if x == Photo.Class.Acceptable.value else 0 for x in actual]
        predicted_binary = [1 if x == Photo.Class.Acceptable.value else 0 for x in predicted]
        prec, recall, _ = metrics.precision_recall_curve(actual_binary, predicted_binary)
        pr_display = metrics.PrecisionRecallDisplay(precision=prec, recall=recall).plot()
        
    @staticmethod
    def plot_roc_graph(actual, predicted):
        actual_binary = [1 if x == Photo.Class.Acceptable.value else 0 for x in actual]
        predicted_binary = [1 if x == Photo.Class.Acceptable.value else 0 for x in predicted]
        fpr, tpr, _ = metrics.roc_curve(actual_binary, predicted_binary)
        # roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        roc_auc = metrics.auc(fpr, tpr)
    
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()
        

    @staticmethod
    def plot_roc_curve(actual, predicted):
        """
        plots the roc curve based of the probabilities
        """
    
        fpr, tpr, thresholds = metrics.roc_curve(actual, predicted, pos_label=Photo.Class.Acceptable.value)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    @staticmethod
    def visualize_photos(photos, grid_size=None):
        """
        Visualize photos in a grid with colors for TP, FP, TN, FN, and Uncategorized.
        Each square will also show the photo ID (extracted from filename).
        """
        photos = Utils.sort_photos(photos)
        # Define color map
        color_map = {
            'TP': 'green',
            'FP': 'red',
            'TN': 'blue',
            'FN': 'orange',
            'Uncategorized': 'gray'
        }
    
        n = len(photos)
        if not grid_size:
            grid_cols = math.ceil(math.sqrt(n))
            grid_rows = math.ceil(n / grid_cols)
        else:
            grid_rows, grid_cols = grid_size
    
        fig, ax = plt.subplots(figsize=(grid_cols, grid_rows))
        ax.set_xlim(0, grid_cols)
        ax.set_ylim(0, grid_rows)
        ax.set_aspect('equal')
        ax.axis('off')
    
        for i, photo in enumerate(photos):
            row = grid_rows - 1 - i // grid_cols
            col = i % grid_cols
            color = color_map.get(photo.case, 'black')
            rect = plt.Rectangle((col, row), 1, 1, color=color)
            ax.add_patch(rect)
    
            # Photo ID from path (assuming filename like '123.jpg' or similar)
            try:
                photo_id = int(''.join(filter(str.isdigit, photo.path)))
            except ValueError:
                photo_id = i + 1  # fallback
    
            ax.text(col + 0.5, row + 0.5, str(photo_id),
                    color='white', ha='center', va='center', fontsize=6, weight='bold')
    
        # Add legend
        legend_patches = [mpatches.Patch(color=color, label=case) for case, color in color_map.items()]
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def sort_photos(photos):
        acc_photos = []
        unacc_photos = []
        for photo in photos:
            if (photo.actual == Photo.Class.Acceptable):
                acc_photos.append(photo)
            else: 
                unacc_photos.append(photo)
        acc_photos = sorted(acc_photos, key=lambda photo: photo.id)
        unacc_photos = sorted(unacc_photos, key=lambda photo: photo.id)
        return acc_photos + unacc_photos

        