import torch
import cv2
import torch.nn.functional as F
from tkinter import filedialog
from torch.nn.modules.distance import PairwiseDistance
from PIL import Image
from models.spoofing.FasNet import Fasnet
import numpy as np
from playsound import playsound
import yaml
import time
import threading
from tkinter import messagebox
import os
from torchvision import transforms
from models.face_detect.mtcnn import MTCNN
from PIL import Image
from PIL import Image, ImageTk
import customtkinter 


# use config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['collect_data']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
antispoof_model = Fasnet()

mtcnn = MTCNN(
    image_size=160, margin=80, min_face_size=20,
    thresholds=[0.8, 0.8, 0.8], factor=0.709, post_process=False,
    select_largest= True,
    selection_method= 'largest',
    device=device,
)



def get_align(image):
    face= None
    input_image = image
    prob = 0
    lanmark = None

    faces, probs, lanmarks = mtcnn.detect(image, landmarks= True)

    if faces is not None and len(faces) > 0:
        face = faces[0] # get highest area bboxes
        prob = probs[0]
        lanmark= lanmarks[0]
        input_image = mtcnn(image)
    return input_image, face, prob, lanmark

def collect_images( 
                canvas, 
                root,
                min_face_area=config['min_face_area'],
                bbox_threshold= config['bbox_threshold'], 
                required_images=config['required_images'],
              ):
 
    global embeddings, image2class, index2class

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera")
        return
    for _ in range(1):
        cap.read()

    valid_images = []
    is_reals = []
    previous_message = 0
    last_sound_time = 0
    sound_delay = 2 

    def update_frame():
        '''
          Continuously captures frames from the camera, detects faces, and validates them.
          
        '''
        nonlocal previous_message, valid_images, is_reals, last_sound_time
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't capture frame")
                break

            input_image, face, prob, landmark = get_align(frame) 

            if face is not None:
                x1, y1, x2, y2 = map(int, face)
                if prob > bbox_threshold:
                    line_length = 20
                    color = (0, 255, 0) 
                    thickness = 1     

                    cv2.line(frame, (x1, y1), (x1 + line_length, y1), color, thickness) 
                    cv2.line(frame, (x1, y1), (x1, y1 + line_length), color, thickness) 

                    cv2.line(frame, (x2, y1), (x2 - line_length, y1), color, thickness) 
                    cv2.line(frame, (x2, y1), (x2, y1 + line_length), color, thickness) 

                    cv2.line(frame, (x1, y2), (x1 + line_length, y2), color, thickness) 
                    cv2.line(frame, (x1, y2), (x1, y2 - line_length), color, thickness)

                    cv2.line(frame, (x2, y2), (x2 - line_length, y2), color, thickness) 
                    cv2.line(frame, (x2, y2), (x2, y2 - line_length), color, thickness) 

                    cv2.putText(frame, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                area = (face[2] - face[0]) * (face[3] - face[1])

                if prob > bbox_threshold:
                    center = np.mean(landmark, axis=0)
                    height, width, _ = frame.shape
                    center_x, center_y = center

                    distance_from_center = np.sqrt((center_x - width / 2) ** 2 + (center_y - height / 2) ** 2)
                    current_time = time.time()

                    if area > min_face_area:
                        if width * 0.15 < center_x < width * 0.85 and height * 0.15 < center_y < height * 0.85 and distance_from_center < min(width, height) * 0.4:
                            if previous_message != 1 and current_time - last_sound_time > sound_delay:
                                threading.Thread(target=playsound, args=('audio/guide_keepface.mp3',), daemon=True).start()
                                last_sound_time = current_time
                                previous_message = 1

                            is_real, score = antispoof_model.analyze(frame, map(int, face)) 
                            print(is_real, score)
                            is_reals.append((is_real, score))
                            valid_images.append(input_image)

                        else:
                            if previous_message != 2 and current_time - last_sound_time > sound_delay:
                                threading.Thread(target=playsound, args=('audio/guide_centerface.mp3',), daemon=True).start()
                                last_sound_time = current_time
                                previous_message = 2

                    else:
                        if previous_message != 3 and current_time - last_sound_time > sound_delay:
                            threading.Thread(target=playsound, args=('audio/closer.mp3',), daemon=True).start()
                            last_sound_time = current_time
                            previous_message = 3

            else:
                if previous_message != 0:
                    print("No face")
                    previous_message = 0

            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image_tk = ImageTk.PhotoImage(image)

            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=image_tk, anchor="center")
            canvas.image = image_tk

            if len(valid_images) >= required_images:
                print(f"Collected enough {required_images} valid images")
                result = {
                    'valid_images': valid_images,
                    'is_reals': is_reals
                }

                save_images(image_list= result['valid_images'], root=root )
                def clear_canvas():
                    canvas.delete('all')
                root.after(3000, clear_canvas)

                break

        cap.release()

    threading.Thread(target=update_frame, daemon=True).start()


def save_images(image_list, root):
    # Tạo cửa sổ mới
    save_folder = filedialog.askdirectory(title="Chọn thư mục để lưu ảnh")

    new_window = customtkinter.CTkToplevel(root)
    new_window.geometry("500x600")
    new_window.title("Data Folder Explorer")

    if not save_folder:
        new_window.destroy()  # Đóng cửa sổ nếu không chọn thư mục
        return

    folder_name_input = customtkinter.CTkTextbox(new_window, height=5)  # Thêm textbox vào cửa sổ mới
    folder_name_input.pack(pady=20)

    sub_folder = ''
    def get_folder_name():
        nonlocal sub_folder
        sub_folder = folder_name_input.get("1.0", "end-1c")  # Lấy tên thư mục

        new_folder_path = os.path.join(save_folder, sub_folder)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        for idx, image in enumerate(image_list):
            save_path = os.path.join(save_folder, f"{sub_folder}/transformed_image_{idx+1}.png")
            image = image.permute(1, 2, 0).numpy()
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path, image)

        messagebox.showinfo("Successful", f"Saved {len(image_list)} images to {new_folder_path}")
        new_window.destroy()  # Đóng cửa sổ sau khi lưu thành công

    button_save = customtkinter.CTkButton(new_window, text="Lưu ảnh", width=110, height=30, command=get_folder_name)
    button_save.pack(pady=7)




