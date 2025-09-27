import os
import shutil
import kaggle

kaggle.api.authenticate()

print(kaggle.api.dataset_list_files('snap/amazon-fine-food-reviews').files)

kaggle.api.dataset_download_files('snap/amazon-fine-food-reviews', 'tempo', unzip=True)

review_csv = './tempo/Reviews.csv'
new_location = os.path.join(review_csv, '../../..')

destination = os.path.abspath(new_location)

shutil.move(review_csv, destination)

deletion_folder = 'tempo'
deletion_path = os.path.join(os.getcwd(), deletion_folder)

shutil.rmtree(deletion_path)
