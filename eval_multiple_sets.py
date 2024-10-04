import os
from pathlib import Path
from eval_ssd import eval_dataset, parser

parser.add_argument("--datasets", nargs="+", help="Dataset directory path")
parser.add_argument("--dataset_dir", nargs="+", help="datasets parent directory")


def parse_datasets(dataset_dir):
    dataset_list = os.listdir(dataset_dir)
    dataset_list = [Path(dataset_dir, subset) for subset in dataset_list]

    return dataset_list


def main():
    args = parser.parse_args()

    dataset_list = args.datasets

    if args.dataset_dir:
        dataset_list = []
        for set_dir in args.dataset_dir:
            # dataset_list.conca parse_datasets(set_dir)
            dataset_list += parse_datasets(set_dir)

    aps = []
    num_gt_per_ds = []

    for dataset in dataset_list:
        print(f"Evaluating {dataset}")
        ap, num_gt = eval_dataset(
            args.eval_dir,
            dataset,
            args.dataset_type,
            args.nms_method,
            args.image_extension,
            args.mb2_width_mult,
            args.net,
            args.trained_model,
            args.iou_threshold,
            args.use_2007_metric,
        )

        num_gt_per_ds.append(num_gt)
        aps.append(ap)

    for dataset, ap in zip(dataset_list, aps):
        if type(dataset) == "str":
            res = dataset.split("\\")
            print(f"{res[-1]}, {ap}")
        else:
            print(f"{dataset.name}, {ap}")

    # Calculate weighted average across all sets:

    total_ap = 0
    total_gt = 0

    for ap, num_gt in zip(aps, num_gt_per_ds):
        total_ap += ap * num_gt
        total_gt += num_gt

    print(f"Weighted AP across all sets: {total_ap/total_gt}")


if __name__ == "__main__":
    main()
