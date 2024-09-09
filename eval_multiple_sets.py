from eval_ssd import eval_dataset, parser

parser.add_argument("--datasets", nargs="+", help="Dataset directory path")


def main():
    args = parser.parse_args()
    datasets = args.datasets
    aps = []

    for dataset in datasets:
        print(f"Evaluating {dataset}")
        ap = eval_dataset(
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

        aps.append(ap)

    for dataset, ap in zip(datasets, aps):
        res = dataset.split("\\")
        print(f"{res[-1]}, {ap}")


if __name__ == "__main__":
    main()
