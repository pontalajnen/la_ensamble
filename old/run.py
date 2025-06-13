import torch
from argparse import ArgumentParser, BooleanOptionalAction
from models.resnet import ResNet18
from train import ResNetTrainer
from dataset import Cifar


def main(args):
    model = ResNet18()

    lr = 0.1
    # should scale with batch size, * bs/256
    momentum = 0.9
    weight_decay = 0.0001
    batch_size = 64
    epochs = 10

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    dataset = Cifar()

    trainer = ResNetTrainer(
        model=model,
        batch_size=batch_size,
        optimizer=optimizer,
        dataset=dataset
    )

    if args.test:
        trainer.model.load_state_dict(torch.load(args.model_path))
        trainer.test()
    else:
        trainer.train(epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    # SEED
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds_per_job", type=int, default=1)

    # Model and Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--basic_augment", action=BooleanOptionalAction, default=True,
                        help="True if you want to use basic augmentations (horizontal flip, random crop with padding).")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Split the training set into train and validation set.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for the dataloader.")
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--depth", default=18, type=int, help="Number of layers.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--model_name", type=str, default="Unknown")
    parser.add_argument("--ViT_model", type=str, default='google/vit-base-patch16-224-in21k',
                        help="Path to checkpoint for fine-tuning")
    parser.add_argument("--normalize_pretrained_dataset", action=BooleanOptionalAction, default=False,
                        help="Finetune the dataset using the normalization values of the pretrained dataset (VIT)")

    # Training
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--SAM", default=False, action=BooleanOptionalAction,
                        help="True if you want to use the SAM optimizer.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler.")
    parser.add_argument("--base_optimizer", type=str, default="SGD", help="Base optimizer.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders.")

    # SAM hyperparameters
    parser.add_argument("--adaptive", default=False, action=BooleanOptionalAction,
                        help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    # parser.add_argument("--rho_min", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--eta", default=0.1, type=float, help="Eta parameter for ASAM.")

    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")

    parser.add_argument("--test", action=BooleanOptionalAction, default=False)
    parser.add_argument("--model_path", type=str, default="saved_models/model_final.pth")

    args = parser.parse_args()
    main(args)
