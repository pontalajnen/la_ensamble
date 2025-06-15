from argparse import ArgumentParser


def common_arguments(parser: ArgumentParser):
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for the dataloader.")
    return parser
