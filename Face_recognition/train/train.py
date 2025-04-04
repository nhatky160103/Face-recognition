import torch 
from torchvision import transforms
from models.face_recogn.inceptionresnetV1 import InceptionResnetV1
from dataloader.Dataset import LFWDataset, TripletFaceDataset
from torch.nn.modules.distance import PairwiseDistance
import torch.optim as optim
import numpy as np
from torch.autograd import Function
from tqdm import tqdm
import os
import torch.nn as nn
from validation import validate_lfw



def set_model_gpu_mode(model):
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu


def set_optimizer(optimizer, model, learning_rate):
    fine_tune_params = [
    {'params': model.last_linear.parameters()},
    {'params': model.last_bn.parameters()},
    ]

    if optimizer == "adagrad":
        optimizer_model = optim.Adagrad(
            params=fine_tune_params,
            lr=learning_rate,
            lr_decay=0,
            initial_accumulator_value=0.1,
            eps=1e-10,
            weight_decay=1e-5
        )


    return optimizer_model

def set_model_architecture(model_architecture, pretrained, embedding_dimension):
    if model_architecture == "InceptionResnetV1":
        model = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None, dropout_prob=0.6, device=device)

    print("Using {} model architecture.".format(model_architecture))

    for param in model.parameters():
        param.requires_grad = False

    for param in model.last_linear.parameters():
        param.requires_grad = True
    for param in model.last_bn.parameters():
        param.requires_grad = True

    return model


class TripletLoss(Function):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss
        

def forward_pass(imgs, model, batch_size):
    
    imgs = imgs.cuda()
    embeddings = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size: batch_size * 2]
    neg_embeddings = embeddings[batch_size * 2:]

    return anc_embeddings, pos_embeddings, neg_embeddings, model



def main():
    dataroot = '/kaggle/input/training-dataset-vn-celeb/VN-celeb'
    lfw_dataroot = '/kaggle/input/data-train-and-val/lfw-224/lfw_224'
    training_dataset_csv_path = '/kaggle/working/training_dataset.csv'
    epochs = 100
    iterations_per_epoch = 200
    model_architecture = 'InceptionResnetV1'
    pretrained = True
    embedding_dimension = 512
    num_human_identities_per_batch = 32
    batch_size = 64
    lfw_batch_size = 32
    resume_path = None
    num_workers = 4
    optimizer = 'adagrad'
    learning_rate = 0.05
    margin = 0.2
    image_size = 160
    use_semihard_negatives =False
    training_triplets_path = None
    flag_training_triplets_path = False
    start_epoch = 0


    os.makedirs('/kaggle/working/plots', exist_ok=True)
    os.makedirs('/kaggle/working/logs', exist_ok=True)
    os.makedirs('/kaggle/working/plots/accuracies_plots', exist_ok=True)
    os.makedirs('/kaggle/working/plots/roc_plots', exist_ok=True)
    os.makedirs('/kaggle/working/model_training_checkpoints', exist_ok= True)
    if training_triplets_path is not None:
        flag_training_triplets_path = True  # Load triplets file for the first training epoch

    # Define image data pre-processing transforms
    #   ToTensor() normalizes pixel values between [0, 1]
    #   Normalize(mean=[0.6071, 0.4609, 0.3944], std=[0.2457, 0.2175, 0.2129]) according to the calculated glint360k
    #   dataset with tightly-cropped faces dataset RGB channels' mean and std values by
    #   calculate_glint360k_rgb_mean_std.py in 'datasets' folder.
    data_transforms = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
           mean =[0.5993, 0.4683, 0.4070], 
           std=[0.2628, 0.2323, 0.2258]
        )
    ])


    lfw_transforms = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(
           mean =[0.485, 0.456, 0.406], 
           std=[0.48, 0.46, 0.44]
        )
    ])

    lfw_dataloader = torch.utils.data.DataLoader(
        dataset=LFWDataset(
            dir=lfw_dataroot,
            pairs_path='/kaggle/input/data-train-and-val/LFW_pairs.txt',
            transform=lfw_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # Instantiate model
    model = set_model_architecture(
        model_architecture=model_architecture,
        pretrained=pretrained,
        embedding_dimension=embedding_dimension
    )

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode(model)

    # Set optimizer
    optimizer_model = set_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate
    )

    # Resume from a model checkpoint
    if resume_path:
        if os.path.isfile(resume_path):
            print("Loading checkpoint {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch'] + 1
            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])

            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            if flag_train_multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded: start epoch from checkpoint = {}".format(start_epoch))
        else:
            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))

    if use_semihard_negatives:
        print("Using Semi-Hard negative triplet selection!")
    else:
        print("Using Hard negative triplet selection!")

    start_epoch = start_epoch

    print("Training using triplet loss starting for {} epochs:\n".format(epochs - start_epoch))

    
    
    
    for epoch in range(start_epoch, epochs):
        num_valid_training_triplets = 0
        l2_distance = PairwiseDistance(p=2)
        _training_triplets_path = None

        if flag_training_triplets_path:
            _training_triplets_path = training_triplets_path
            flag_training_triplets_path = False  # Only load triplets file for the first epoch

        print('start gen triplet')
        # Re-instantiate training dataloader to generate a triplet list for this training epoch
        train_dataloader = torch.utils.data.DataLoader(
            dataset=TripletFaceDataset(
                root_dir=dataroot,
                training_dataset_csv_path=training_dataset_csv_path,
                num_triplets=iterations_per_epoch * batch_size,
                num_human_identities_per_batch=num_human_identities_per_batch,
                triplet_batch_size=batch_size,
                epoch=epoch,
                training_triplets_path=_training_triplets_path,
                transform=data_transforms
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False  # Shuffling for triplets with set amount of human identities per batch is not required
        )

        print('start training')

        # Training pass
        model.train()
        progress_bar = enumerate(tqdm(train_dataloader))

        for batch_idx, (batch_sample) in progress_bar:

            # Forward pass - compute embeddings
            anc_imgs = batch_sample['anc_img']
            pos_imgs = batch_sample['pos_img']
            neg_imgs = batch_sample['neg_img']

            
            # Concatenate the input images into one tensor because doing multiple forward passes would create
            #  weird GPU memory allocation behaviours later on during training which would cause GPU Out of Memory
            #  issues
            all_imgs = torch.cat((anc_imgs, pos_imgs, neg_imgs))  # Must be a tuple of Torch Tensors

            anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                imgs=all_imgs,
                model=model,
                batch_size=batch_size
            )

            pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
            neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)

            if use_semihard_negatives:
                # Semi-Hard Negative triplet selection
                #  (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
                #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L295
                first_condition = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                second_condition = (pos_dists < neg_dists).cpu().numpy().flatten()
                all = (np.logical_and(first_condition, second_condition))
                valid_triplets = np.where(all == 1)
            else:
                # Hard Negative triplet selection
                #  (negative_distance - positive_distance < margin)
                #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L296
                all = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                valid_triplets = np.where(all == 1)

            anc_valid_embeddings = anc_embeddings[valid_triplets]
            pos_valid_embeddings = pos_embeddings[valid_triplets]
            neg_valid_embeddings = neg_embeddings[valid_triplets]

            # Calculate triplet loss
            triplet_loss = TripletLoss(margin=margin).forward(
                anchor=anc_valid_embeddings,
                positive=pos_valid_embeddings,
                negative=neg_valid_embeddings
            )

            # Calculating number of triplets that met the triplet selection method during the epoch
            num_valid_training_triplets += len(anc_valid_embeddings)

            # Backward pass
            optimizer_model.zero_grad()
            triplet_loss.backward()
            optimizer_model.step()

        # Print training statistics for epoch and add to log
        print('Epoch {}:\tNumber of valid training triplets in epoch: {}'.format(
                epoch,
                num_valid_training_triplets
            )
        )

        with open('/kaggle/working/logs/{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch,
                num_valid_training_triplets
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

        # Evaluation pass on LFW dataset
        best_distances = validate_lfw(
            model=model,
            lfw_dataloader=lfw_dataloader,
            model_architecture=model_architecture,
            epoch=epoch
        )

        # Save model checkpoint
        state = {
            'epoch': epoch,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'best_distance_threshold': np.mean(best_distances)
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()

        # Save model checkpoint
        torch.save(state, 'model_training_checkpoints/model_{}_triplet_epoch_{}.pt'.format(
                model_architecture,
                epoch
            )
        )