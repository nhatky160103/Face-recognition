from infer.infer_video import check_validation
from infer.infer_image import get_align, infer
from infer.identity_person import find_closest_person, find_closest_person_vote
from infer.get_embedding import load_embeddings_and_names
from torchvision import datasets
from infer.utils import get_model
from torch.utils.data import DataLoader
from infer.getface import mtcnn_inceptionresnetV1
from infer.getface import yolo
from infer.infer_image import inceptionresnetV1_transform
import os
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(x):
        return x[0]


def test_model(test_folder, recognition_model_name, embedding_file_path, image2class_file_path, index2class_file_path, batch_size=512):
    '''
    Tests a face recognition model by comparing embeddings from test images with pre-stored embeddings.

    Parameters:
    - test_folder (str): Path to the folder containing test images organized by class.
    - recognition_model_name (str): Name of the pre-trained recognition model to be loaded.
    - embedding_file_path (str): Path to the file containing stored embeddings for known identities.
    - image2class_file_path (str): Path to the file mapping image indices to class IDs.
    - index2class_file_path (str): Path to the file mapping class indices to class names.
    - batch_size (int, optional): Number of images to process in a single batch. Default is 512.

    Returns:
    - float: Percentage of correctly matched test images based on the model's predictions.

    '''
    recognition_model = get_model(recognition_model_name)

   

    dataset = datasets.ImageFolder(test_folder)
    dataset.index2class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    
    aligned = []  # List of aligned images in batches
    image2class_test = {}

    for i, (x, y) in enumerate(loader):
        image2class_test[i] = y
        x_aligned = mtcnn_inceptionresnetV1(x)
    
        if x_aligned is not None:
            x_aligned = inceptionresnetV1_transform(x_aligned)
            aligned.append(x_aligned)
        else:
            results = yolo(x)
            if results[0].boxes.xyxy.shape[0] != 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, boxes[0])
                face = x.crop((x1, y1, x2, y2)).resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = inceptionresnetV1_transform(face)
                aligned.append(x_aligned)
            else:
                print('No detect face, get full image')
                face = x.resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = inceptionresnetV1_transform(face)
                aligned.append(x_aligned)

        # Process batch when it reaches the batch_size
        if len(aligned) >= batch_size:
            batch = torch.cat(aligned[:batch_size], dim=0).to(device)
            batch_embeddings = recognition_model(batch).detach().cpu()
            if 'test_embeddings' not in locals():
                test_embeddings = batch_embeddings
            else:
                test_embeddings = torch.cat((test_embeddings, batch_embeddings), dim=0)

            # Remove processed items
            aligned = aligned[batch_size:]

    # Process remaining items in the aligned list
    if aligned:
        batch = torch.cat(aligned, dim=0).to(device)
        batch_embeddings = recognition_model(batch).detach().cpu()
        if 'test_embeddings' not in locals():
            test_embeddings = batch_embeddings
        else:
            test_embeddings = torch.cat((test_embeddings, batch_embeddings), dim=0)

    embeddings, image2class, index2class = load_embeddings_and_names(
        embedding_file_path, image2class_file_path, index2class_file_path
    )

    class_ids = []
    for i, test_embedding in enumerate(test_embeddings):
        class_id = find_closest_person(
            test_embedding, 
            embeddings, 
            image2class
        )
        class_ids.append((class_id, image2class_test[i]))

    print(len(class_ids))
    print(class_ids)

    matching_elements = sum(1 for item in class_ids if item[0] == item[1])
    total_elements = len(class_ids)
    percentage = (matching_elements / total_elements) * 100

    return percentage

def plot_with_images(cosine_sim_matrix, folder_1_images, folder_2_images, image_size=(65, 65), zoom=0.5):

    cosine_sim_matrix = cosine_sim_matrix[np.argsort(folder_1_images)][:, np.argsort(folder_2_images)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cosine_sim_matrix, annot=False, cmap='coolwarm', cbar=True, ax=ax)

    for y, img_path in enumerate(folder_1_images):
        img = Image.open(img_path).resize(image_size)
        img = np.array(img)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (-0.5, y + 0.5), frameon=False, box_alignment=(1, 0.5))
        ax.add_artist(ab)

    for x, img_path in enumerate(folder_2_images):
        img = Image.open(img_path).resize(image_size)
        img = np.array(img)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x + 0.5, cosine_sim_matrix.shape[0] + 0.5), frameon=False, box_alignment=(0.5, 0))
        ax.add_artist(ab)

    # Hiển thị giá trị cosine similarity trên heatmap
    for y in range(cosine_sim_matrix.shape[0]):
        for x in range(cosine_sim_matrix.shape[1]):
            score = f"{cosine_sim_matrix[y, x]:.2f}"
            ax.text(x + 0.5, y + 0.5, score, ha='center', va='center', color='black', fontsize=8)

    # Đảm bảo vùng hiển thị
    ax.set_xlim(-0.5, cosine_sim_matrix.shape[1] + 0.5)
    ax.set_ylim(-0.5, cosine_sim_matrix.shape[0] + 0.5)

    plt.xlabel('Images in Folder 2', fontsize=12)
    plt.ylabel('Images in Folder 1', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.show()



def test_model_facerecognmodel(test_folder_1, test_folder_2, recognition_model_name, device='cpu'):
    recognition_model = get_model(recognition_model_name).to(device)
    recognition_model.eval() 
    
    dataset_1 = datasets.ImageFolder(test_folder_1)
    dataset_2 = datasets.ImageFolder(test_folder_2)

    dataset_1.index2class = {i: c for c, i in dataset_1.class_to_idx.items()}
    dataset_2.index2class = {i: c for c, i in dataset_2.class_to_idx.items()}

    print(dataset_1.index2class)
    print(dataset_2.index2class)

    loader_1 = DataLoader(dataset_1, collate_fn=collate_fn, num_workers=workers)
    loader_2 = DataLoader(dataset_2, collate_fn=collate_fn, num_workers=workers)

    aligned_1 = []  # List of aligned images from folder 1
    aligned_2 = []  # List of aligned images from folder 2

    with torch.no_grad():  
        for x, _ in loader_1:
            x_aligned = mtcnn_inceptionresnetV1(x)
            if x_aligned is not None:
                x_aligned = inceptionresnetV1_transform(x_aligned)
            x_aligned = x_aligned.to(device)
            embeddings = recognition_model(x_aligned)
            print('embeddings')
            aligned_1.append(embeddings)
        print('len align 1:', len(aligned_1))

    with torch.no_grad():
        for x, _ in loader_2:
            x_aligned = mtcnn_inceptionresnetV1(x)
            if x_aligned is not None:
                x_aligned = inceptionresnetV1_transform(x_aligned)
            x_aligned = x_aligned.to(device)
            embeddings = recognition_model(x_aligned)
            aligned_2.append(embeddings)
        print('len align 2:', len(aligned_2))

    embeddings_1 = torch.cat(aligned_1, dim=0).cpu().numpy()
    embeddings_2 = torch.cat(aligned_2, dim=0).cpu().numpy()

    print('embeddings1:', embeddings_1.shape)
    print('embeddings2:', embeddings_2.shape)

    cosine_sim_matrix = cosine_similarity(embeddings_1, embeddings_2)

    folder_1_images = [dataset_1.imgs[i][0] for i in range(len(dataset_1))]
    folder_2_images = [dataset_2.imgs[i][0] for i in range(len(dataset_2))]

    plot_with_images(cosine_sim_matrix, folder_1_images, folder_2_images)

    return cosine_sim_matrix


if __name__ == "__main__":
    # test_folder = 'data/data_gallery_2'

    # embedding_file_path= 'data/data_source/db2/inceptionresnetV1_embeddings.npy'
    # image2class_file_path = 'data/data_source/db2/inceptionresnetV1_image2class.pkl'
    # index2class_file_path = 'data/data_source/db2/inceptionresnetV1_index2class.pkl'

    # percentage = test_model(test_folder=test_folder, 
    #            recognition_model_name='inceptionresnetV1',
    #            embedding_file_path=embedding_file_path,
    #            image2class_file_path=image2class_file_path,
    #            index2class_file_path= index2class_file_path,
    #            )
    
    # print(percentage)

    folder1 = 'data/folder1'
    folder2 = 'data/folder2'
    cosine_sim_matrix = test_model_facerecognmodel(folder1, folder2,  'inceptionresnetV1')
    print(cosine_sim_matrix.shape)