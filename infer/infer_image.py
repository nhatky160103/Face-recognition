import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_model
from PIL import Image
import torch
from torchvision import transforms
from .getface import mtcnn_inceptionresnetV1, yolo
import torch.nn.functional as F
import cv2




# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define transform
def inceptionresnetV1_transform(img):
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    return img


# infer function
def infer(recogn_model, align_image):
    '''
     Runs inference on an aligned image to generate its embedding using the recognition model.

    Parameters:
    ----------
    recogn_model : torch.nn.Module
        The face recognition model used to generate embeddings.
    align_image : PIL.Image.Image
        The pre-aligned input image for which the embedding is to be generated.

    Returns:
    -------
    embedding : torch.Tensor or None
        The embedding vector for the input image. Returns `None` if an error occurs. 
    '''
    try:
        input_image = inceptionresnetV1_transform(align_image)
        embedding = recogn_model(input_image)
        return embedding

    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    


def get_align(image):
    '''
    Aligns and extracts facial information from an input image using MTCNN and InceptionResNetV1.

    Parameters:
        image (PIL.Image or ndarray): The input image containing a face.

    Returns:
        tuple:
            input_image (torch.Tensor or None): The preprocessed image tensor of the detected face if found, otherwise the original image.
            face (ndarray or None): The bounding box coordinates (x_min, y_min, x_max, y_max) of the detected face if found, otherwise None.
            prob (float): The probability/confidence score of the detected face. Default is 0 if no face is detected.
            lanmark (ndarray or None): The landmarks (e.g., eyes, nose, mouth corners) of the detected face if found, otherwise None.

    '''
    face= None
    input_image = image
    prob = 0
    lanmark = None

    faces, probs, lanmarks = mtcnn_inceptionresnetV1.detect(image, landmarks= True)

    if faces is not None and len(faces) > 0:
        face = faces[0] # get highest area bboxes
        prob = probs[0]
        lanmark= lanmarks[0]
        input_image = mtcnn_inceptionresnetV1(image)

    return input_image, face, prob, lanmark


if __name__ == "__main__":
    
    image = Image.open('data/data_gallery_1/Dinh Nhat Ky/transformed_image_21.png').convert('RGB')
    input_image, face, prob, landmark = get_align(image)

    plt.imshow(input_image.permute(1, 2, 0).numpy())
    plt.show()

    print(input_image.shape)
    print(face)
    print(prob)
    print(landmark)

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = map(int, face)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(image, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in landmark:
        cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)


    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
