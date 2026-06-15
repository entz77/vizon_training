
import json
import os
import random
from google.cloud import storage

def split_data(gcs_annotation_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42, gcs_default_path='gs://dev-vveesee-ml-training/vizon-training/manual_annotations/detector/'):
    """
    1. List all annotation files in the local folder.
    2. Shuffle the list of files using the provided random seed.
    3. Split the list of files into training, validation, and test sets based on the provided ratios.
    4. Return three lists: one for training files, one for validation files, and one for test files.

    Args:
        train_ratio (float): The ratio of files to be used for training. Default is 0.8.
        val_ratio (float): The ratio of files to be used for validation. Default is 0.1.
        test_ratio (float): The ratio of files to be used for testing. Default is 0.1.
        random_seed (int): The random seed to be used for shuffling the files. Default is 42.
        gcs_annotation_folder (str): The GCS folder containing the annotation files.
        gcs_default_path (str): The default GCS path for the annotation files.
    """

    # List all annotation files in a gcs folder
    full_path = gcs_default_path.rstrip('/') + '/' + gcs_annotation_folder.strip('/')
    print(full_path)
    client = storage.Client()
    bucket_name = full_path.split('/')[2]
    bucket = client.bucket(bucket_name)
    prefix = '/'.join(full_path.split('/')[3:]) + '/'
    all_files = [f'gs://{bucket_name}/' + blob.name for blob in bucket.list_blobs(prefix=prefix) if blob.name.endswith('.txt')]

    # Shuffle the list of files using the provided random seed
    random.seed(random_seed)
    random.shuffle(all_files)

    # Calculate the number of files for each set
    total_files = len(all_files)
    train_count = int(total_files * train_ratio)
    if test_ratio == 0:
        val_count = total_files - train_count
    else:
        val_count = int(total_files * val_ratio)

    # Split the list of files into training, validation, and test sets
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    return train_files, val_files, test_files

def summarize_data(train_files, val_files, test_files, gcs_destination_folder):
    """Given a GCS source folder with annotation data (TXT files) and gcs destination folder,
    1. Infer the image file names
    2. Create the destination blob names for the annotation files and the corresponding image files in the destination folder.
    3. Summarize the data and return a list of tuples containing the source file name and destintion blob name for the annotation files and the corresponding image files in the destination folder.
    for the training, validation, and test sets.
    """
    data = []
    cls_data = {}
    image_src_base_path = 'gs://dev-vveesee-edge-submit/images/'
    dest_base_path = 'gs://dev-vveesee-ml-training/vizon-training/datasets/'
    for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in split_files:
            image_file_name = file.replace('.txt', '.jpg').split('/')[-1]
            image_file_name = image_src_base_path + image_file_name
            annotation_dest = os.path.join(dest_base_path, gcs_destination_folder, 'labels', split_name, os.path.basename(file))
            image_dest = os.path.join(dest_base_path, gcs_destination_folder, 'images', split_name, os.path.basename(image_file_name))
            data.append((file, annotation_dest))
            data.append((image_file_name, image_dest))
            cls_data[image_file_name] = file

    return data, cls_data

def copy_data_in_gcs(data):
    """
    1. Given list of tuples containing the source file name and destintion blob name for the annotation files
    and the corresponding image files in the destination folder, copy the files from the source location
    to the destination location in GCS.
    2. Dump data to a file in the source folder
    """
    client = storage.Client()
    for source, destination in data:
        source_bucket_name = source.split('/')[2]
        source_blob_name = '/'.join(source.split('/')[3:])
        destination_bucket_name = destination.split('/')[2]
        destination_blob_name = '/'.join(destination.split('/')[3:])

        source_bucket = client.bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_blob_name)

        destination_bucket = client.bucket(destination_bucket_name)

        source_bucket.copy_blob(source_blob, destination_bucket, new_name=destination_blob_name)
        print(f'Successful copy to {destination}')

def dump_data_map(data_map):
    """
    From cls_data dictionary, dump it to as JSON file in the same folder of the first element's value in the dict
    """
    first_value = next(iter(data_map.values()))
    base_folder = '/'.join(first_value.split('/')[:-2])

    bucket_name = base_folder.split('/')[2]
    blob_name = '/'.join(base_folder.split('/')[3:]) + '/det_data_map.json'

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data_map, indent=2), content_type='application/json')

    print('Data for CLS annotation is successfully dumped.')