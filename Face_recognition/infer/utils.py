import torch
import torch.nn as nn
from models.face_recogn.inceptionresnetV1 import InceptionResnetV1

# use device
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_name):
    '''
    Initializes and returns a deep learning model based on the provided model name.

    Parameters:
        model_name (str): The name of the model to load. Currently supports 'inceptionresnetV1'.

    Returns:
        torch.nn.Module: The initialized model set to evaluation mode.

    '''
  
  
     
    model = None
    if model_name == 'inceptionresnetV1':
        model = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None, dropout_prob=0.6, device=device)
        state_dict = None
    else:
        print('please enter correct model! ')

    model, _ = set_model_gpu_mode(model)

    if state_dict:
        model.load_state_dict(state_dict)
    model.eval()
    
    return model

def set_model_gpu_mode(model):
    '''
    Configures the model to run on GPU if available, supporting multi-GPU setups if applicable.

    Parameters:
        model (torch.nn.Module): The PyTorch model to configure for GPU usage.

    Returns:
        tuple: 
            - torch.nn.Module: The model configured for single or multi-GPU usage.
            - bool: A flag indicating if multi-GPU mode is active.
    '''
    
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
