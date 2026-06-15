import argparse
from datasets import preparation

parser = argparse.ArgumentParser()
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--val-ratio', type=float, default=0.3)
parser.add_argument('--test-ratio', type=float, default=0.0)
parser.add_argument('--random-seed', type=int, default=42)
parser.add_argument('--gcs-annotation-folder', help="Folder under gs://dev-vveesee-ml-training/vizon-training/manual_annotations/detector/, unique by Label-Studio Task", type=str, required=True)
parser.add_argument('--gcs-destination-folder', help="Folder under gs://dev-vveesee-ml-training/vizon-training/datasets/, e.g. rtd", type=str, required=True)
args = parser.parse_args()

split_files = preparation.split_data(args.gcs_annotation_folder, args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed)
data, data_map = preparation.summarize_data(*split_files, args.gcs_destination_folder)
preparation.copy_data_in_gcs(data)
preparation.dump_data_map(data_map, args.gcs_annotation_folder)