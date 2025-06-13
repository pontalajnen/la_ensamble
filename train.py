from argparse import ArgumentParser, BooleanOptionalAction
import torch
from torchvision.transforms import v2
import os
import wandb
import timm
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from utils.data import load_data_module
from utils.sam import SAM, enable_running_stats, disable_running_stats
from utils.eval import *
from utils.eval import evaluate_model
from models.resnet import *
from models.resnet import torch_resnet18
import torch.optim as optim
from utils.paths import *
from utils.paths import LOCAL_STORAGE, DATA_DIR, MODEL_PATH_LOCAL
from transformers import ViTImageProcessor  # , ViTForImageClassification


def train(args):

    # Set device
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else  # TODO: Not sure if ":0" is needed
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    print("Device:", device)

    # AVOID WANDB TIMEOUT
    os.environ['WANDB_INIT_TIMEOUT'] = '800'

    # Set path to datasets
    DATA_PATH = LOCAL_STORAGE + DATA_DIR
    # os.makedirs(DATA_PATH, exist_ok=True)

    # Load the dataset
    dm, num_classes = load_data_module(args.dataset, DATA_PATH, batch_size=args.batch_size,
                                       num_workers=args.num_workers, val_split=args.val_split,
                                       basic_augment=args.basic_augment)

    # Resize the images so that it is compatible with a ViT pretrained on ImageNet-21k
    if args.model == "ViT":
        model_name = 'google/vit-base-patch16-224-in21k'
        processor = ViTImageProcessor.from_pretrained(model_name)
        image_mean, image_std = processor.image_mean, processor.image_std
        # size = processor.size["height"]
        if args.normalize_pretrained_dataset:
            normalize = v2.Normalize(mean=image_mean, std=image_std)
        else:
            if args.dataset == "CIFAR10":
                normalize = v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
            elif args.dataset == "CIFAR100":
                normalize = v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

        # resize_transform = v2.Resize((224, 224))

        dm.train_transform = v2.Compose([
            v2.Resize(256),
            v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        dm.test_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    images, labels = next(iter(train_loader))
    print(images.shape)

    print("Successfully loaded the dataset!")
    wandb.login()

    for i in range(args.seeds_per_job):
        print(i, args.seed)
        seed = args.seed + i
        model_name = args.model+"_"+args.dataset+"_seed"+str(seed)+"_"+args.base_optimizer
        torch_seed = torch.Generator()
        torch_seed.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        project = f"LA_SAM_{args.dataset}_{args.model}_SAM{args.SAM}_adaptive{args.adaptive}"
        # Initialize W&B run and log hyperparameters
        run = wandb.init(project=project, name=model_name, config={
            "base_optimizer": args.base_optimizer,
            "rho": args.rho,
            "adaptive": args.adaptive,
            "lr": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "seed": seed,
            "SAM": args.SAM,
            "momentum": args.momentum,
            "epochs": args.epochs,
            "dataset": args.dataset,
            "model": args.model
        })

        # Prepare for saving model checkpoints locally and log them to W&B
        save_dir = MODEL_PATH_LOCAL + f"{args.dataset}_{args.model}_{'' if args.SAM else 'no'}_SAM/"
        os.makedirs(save_dir, exist_ok=True)

        # PATH = (f"{save_dir}_{args.base_optimizer}_rho{args.rho}_adaptive{args.adaptive}"
        # f"_seed{seed}_normorig{args.normalize_pretrained_dataset}_runID{run.id}")
        artifact = wandb.Artifact("model_checkpoints", type="model")

        print("Successfully initialized W&B run!")
        print("i still same?", i, args.seed)
        if args.model == "ResNet18":
            print("Start loading model!")
            model = torch_resnet18(num_classes=num_classes)
            print("Model returned!")
            model = model.to(device)
            print("Model on device!")
        elif args.model == "ViT":
            print(dm.train.dataset.classes)
            model = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True, num_classes=num_classes)
            model = model.to(device)
            print("Model ready and on device!")
        else:
            raise Exception("Oops, requested model does not exist! Model has to be one of 'ResNet18', 'ViT'")

        # Optimizer
        if args.base_optimizer == "SGD":
            base_optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                       weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.base_optimizer == "AdamW":
            base_optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise Exception("Oops, requested optimizer does not exist! Optimizer has to be one of 'SGD'")

        print("Optimizer ready!")
        print("i still same?", i, args.seed)

        # Determine the base optimizer and SAM optimizer setup
        if args.SAM:
            print("Using SAM optimizer!")
            if args.adaptive:
                print("Using Adaptive SAM optimizer!")
            # Set up arguments for both SAM and the base optimizer
            optimizer_args = {
                'params': model.parameters(),
                'base_optimizer': type(base_optimizer),
                'rho': args.rho,
                'adaptive': args.adaptive,
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay,
            }
            if isinstance(base_optimizer, optim.SGD):
                optimizer_args['momentum'] = args.momentum

            # Create the SAM optimizer
            opt = SAM(**optimizer_args)

            # Create the learning rate scheduler for SAM
            if args.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer, args.epochs)
        else:
            # Use the base optimizer without SAM
            opt = base_optimizer
            if args.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

        # Loss function
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        best_epoch = 0
        best_checkpoint_path = os.path.join(save_dir, f"model_{args.model}_seed{seed}_best.pth")
        print("Successfully initialized model, optimizer and loss function!")
        print("Start training loop!")
        for epoch in tqdm(range(args.epochs), desc="Epochs"):
            # Train as usual
            model.train()

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                if args.SAM:
                    # --------------- SAM ----------------
                    # first forward-backward step
                    enable_running_stats(model)  # <- this is the important line
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    loss.mean().backward()
                    opt.first_step(zero_grad=True)

                    # second forward-backward step
                    disable_running_stats(model)  # <- this is the important line
                    criterion(model(x), y).mean().backward()
                    opt.second_step(zero_grad=True)
                    # ------------------------------------
                else:
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    loss.backward()
                    # total_norm = 0.0
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         param_norm = param.grad.data.norm(2)
                    #         #print(f"{name}: grad norm = {param_norm.item()}")
                    #         total_norm += param_norm.item() ** 2

                    # total_norm = total_norm ** 0.5
                    # print(f"Total gradient norm: {total_norm}")
                    opt.step()
                    opt.zero_grad()

            # Validation loop
            val_accuracy, val_loss = evaluate_model(model, val_loader, device, criterion)
            print("i still same?", i, args.seed)
            # Log validation accuracy
            wandb.log({"epoch": epoch, "val_accuracy": val_accuracy,
                       "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})

            # **Save and track the best model**
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_checkpoint_path)  # Overwrites temp file

            scheduler.step()
            print("Finished epoch", epoch)
            print("i still same?", i, args.seed)

        print("Finished training loop!")
        # **Rename the best checkpoint with metadata**
        final_checkpoint_path = os.path.join(
            save_dir,
            (f"seed={seed}-epoch={best_epoch:02d}-val_loss={best_val_loss:.4f}-model={args.model}-"
             f"optimizer={args.base_optimizer}-rho={args.rho}-adaptive={args.adaptive}-model_name={model_name}.pth")
        )
        os.rename(best_checkpoint_path, final_checkpoint_path)  # Rename the best model file

        last_epoch_checkpoint_path = os.path.join(
            save_dir,
            (f"seed={seed}-epoch={args.epochs}-val_loss={val_loss:.4f}-model={args.model}-"
             f"optimizer={args.base_optimizer}-rho={args.rho}-adaptive={args.adaptive}-model_name={model_name}.pth")
        )
        # Store model after last epoch
        torch.save(model.state_dict(), last_epoch_checkpoint_path)

        artifact.add_file(final_checkpoint_path)
        wandb.log_artifact(artifact)
        wandb.finish()


def main():
    parser = ArgumentParser()
    # SEED
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds_per_job", type=int, default=1)

    # Model and Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--basic_augment", action=BooleanOptionalAction, default=True,
                        help="Enable basic augmentations (horizontal flip, random crop with padding).")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Split the training set into train and validation set.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for the dataloader.")
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--depth", default=18, type=int,
                        help="Number of layers.")
    parser.add_argument("--width_factor", default=8, type=int,
                        help="How many times wider compared to normal ResNet.")
    parser.add_argument("--model_name", type=str, default="Unknown")
    parser.add_argument("--ViT_model", type=str, default='google/vit-base-patch16-224-in21k',
                        help="Path to checkpoint for fine-tuning")
    parser.add_argument("--normalize_pretrained_dataset", action=BooleanOptionalAction, default=False,
                        help="Finetune the dataset using the normalization values of the pretrained dataset (VIT)")

    # Training
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=200, type=int,
                        help="Total number of epochs.")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="Dropout rate.")
    parser.add_argument("--SAM", default=False, action=BooleanOptionalAction,
                        help="Enable SAM optimizer.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        help="Learning rate scheduler.")
    parser.add_argument("--base_optimizer", type=str, default="SGD",
                        help="Base optimizer.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                        help="L2 weight decay.")
    parser.add_argument("--threads", default=8, type=int,
                        help="Number of CPU threads for dataloaders.")

    # SAM hyperparameters
    parser.add_argument("--adaptive", default=False, action=BooleanOptionalAction,
                        help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=0.05, type=float,
                        help="Rho parameter for SAM.")
    # parser.add_argument("--rho_min", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=float,
                        help="Rho parameter for SAM.")
    parser.add_argument("--eta", default=0.1, type=float,
                        help="Eta parameter for ASAM.")

    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Use 0.0 for no label smoothing.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
