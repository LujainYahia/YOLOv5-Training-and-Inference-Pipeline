# YOLOv5-Training-and-Inference-Pipeline
Here is a detailed README file for the provided code. This README outlines the setup, usage, and functionalities of the script for training and inference using YOLOv5.

---

# YOLOv5 Training and Inference Pipeline

This repository contains a script to train a YOLOv5 model and perform inference on a set of images. The script includes downloading the dataset, setting up the environment, training the model, and running inference on new images.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Validation](#validation)
- [Inference](#inference)
- [Visualization](#visualization)
- [License](#license)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   ```

2. **Install the Dependencies**

   ```bash
   pip3 install -r requirements.txt
   ```

## Dataset Preparation

1. **Download the Dataset**

   The dataset is downloaded from a public source and unzipped. If the dataset directory already exists, it skips this step.

   ```python
   if not os.path.exists('train'):
       !/usr/bin/curl -L "https://public.roboflow.com/ds/xKLV14HbTF?key=aJzo7msVta" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
       dirs = ['train', 'valid', 'test']
       for i, dir_name in enumerate(dirs):
           all_image_names = sorted(os.listdir(f"{dir_name}/images/"))
           for j, image_name in enumerate(all_image_names):
               if (j % 2) == 0:
                   file_name = image_name.split('.jpg')[0]
                   os.remove(f"{dir_name}/images/{image_name}")
                   os.remove(f"{dir_name}/labels/{file_name}.txt")
   ```

2. **Remove Unnecessary Files**

   The script removes every second image and its corresponding label file to balance the dataset.

## Training

1. **Set Results Directory**

   The `set_res_dir` function sets up the directory to store the training results.

   ```python
   def set_res_dir():
       res_dir_count = len(glob.glob('runs/train/*'))
       print(f"Current number of result directories: {res_dir_count}")
       if TRAIN:
           RES_DIR = f"results_{res_dir_count+1}"
           print(RES_DIR)
       else:
           RES_DIR = f"results_{res_dir_count}"
       return RES_DIR

   RES_DIR = set_res_dir()
   ```

2. **Train the Model**

   The model is trained using the specified number of epochs and batch size. 

   ```python
   if TRAIN:
       !python3 train.py --data ../data.yaml --weights yolov5s.pt \
       --img 640 --epochs {EPOCHS} --batch-size 16 --name {RES_DIR}
   ```

## Validation

The `show_valid_results` function displays the validation results.

```python
def show_valid_results(RES_DIR):
    !ls runs/train/{RES_DIR}
    EXP_PATH = f"runs/train/{RES_DIR}"
    validation_pred_images = glob.glob(f"{EXP_PATH}/*_pred.jpg")
    print(validation_pred_images)
    for pred_image in validation_pred_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
        plt.show()

show_valid_results(RES_DIR)
```

## Inference

1. **Download Inference Data**

   The inference data is downloaded and unzipped if not already present.

   ```python
   download_file('https://learnopencv.s3.us-west-2.amazonaws.com/yolov5_inference_data.zip', 'inference_data.zip')
   if not os.path.exists('inference_images'):
       !unzip -q "inference_data.zip"
   else:
       print('Dataset already present')
   ```

2. **Run Inference**

   The `inference` function runs inference on the provided data path.

   ```python
   def inference(RES_DIR, data_path):
       infer_dir_count = len(glob.glob('runs/detect/*'))
       print(f"Current number of inference detection directories: {infer_dir_count}")
       INFER_DIR = f"inference_{infer_dir_count+1}"
       print(INFER_DIR)

       !python3 detect.py --weights runs/train/{RES_DIR}/weights/best.pt \
       --source {data_path} --name {INFER_DIR}
       return INFER_DIR

   IMAGE_INFER_DIR = inference(RES_DIR, 'inference_images')
   ```

## Visualization

The `visualize` function displays the inference results.

```python
def visualize(INFER_DIR):
    INFER_PATH = f"runs/detect/{INFER_DIR}"
    infer_images = glob.glob(f"{INFER_PATH}/*.jpg")
    print(infer_images)
    for pred_image in infer_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
        plt.show()

visualize(IMAGE_INFER_DIR)
```


---
