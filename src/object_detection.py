import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.modules.upsampling import Upsample
from torchvision.models import detection
from torchvision.transforms import ToTensor

import reproducibility


# https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
class FRCNNDetector(torch.nn.Module):
    def __init__(self, device: str, num_positions: int, roi_pool_size: int):
        """

        :param device:
        :param num_positions: max number of cnsidered objects for both the location
        features and the roi features.
        :param roi_pooled_size:
        """
        super().__init__()
        self.frcnn = detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True).to(device)
        self.frcnn.eval()
        self.num_positions = num_positions
        self.roi_pool_size = roi_pool_size
        self.image_up_or_down_sampler = Upsample(size=roi_pool_size, mode="nearest")
        self.image_to_tensor = ToTensor()
        self.labels = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detections(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)  # batch dim
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to("cuda")
        return self.frcnn(image)[0]

    def extract_location_features(self, image_path: str):
        """
        features = relative bounding box position and size, as well as object label and confidence

        :param image_path:
        :return: features, attention_mask
        """
        image = cv2.imread(image_path)
        detections = self.detections(image)
        width = image.shape[0]
        height = image.shape[1]
        location_features = torch.zeros([self.num_positions, 7])
        roi_features = torch.zeros(
            [self.num_positions, 3, self.roi_pool_size, self.roi_pool_size])  # flattened roi features
        next_to_fill_pos = 0

        for i in range(0, len(detections["boxes"])):
            if next_to_fill_pos == self.num_positions:
                break

            confidence = detections["scores"][i]
            if confidence > 0.9:
                box = detections["boxes"][i].detach().cpu().numpy()
                label = detections["labels"][i]
                (x1, y1, x2, y2) = box.astype("int")
                normed_x1 = x1 / width
                normed_y1 = y1 / height
                normed_x2 = x2 / width
                normed_y2 = y2 / height
                normed_size = (x2 - x1) * (y2 - y1) / (width * height)
                feature = torch.tensor([normed_x1, normed_y1, normed_x2, normed_y2, normed_size, label, confidence])
                location_features[next_to_fill_pos] = feature
                roi_features[next_to_fill_pos] = self.roi_pool(image=image, x1=x1, y1=y1, x2=x2, y2=y2)
                next_to_fill_pos += 1
        attention_mask = torch.ones(self.num_positions)
        attention_mask[next_to_fill_pos:] = attention_mask[next_to_fill_pos:] * 0  # mask out
        return (location_features, attention_mask, roi_features)

    def display_bounding_boxes(self, image_path: str, destination_path: str):
        image = cv2.imread(image_path)
        detections = self.detections(image=image)

        for i in range(0, len(detections["boxes"])):
            confidence = detections["scores"][i]

            if confidence > 0.8:
                idx = int(detections["labels"][i])
                box = detections["boxes"][i].detach().cpu().numpy()
                (start_x, start_y, end_x, end_y) = box.astype("int")
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(0, 0, 255), thickness=2)
                y = start_y - 15 if start_y - 15 > 15 else start_y + 15
                cv2.putText(image, self.labels[idx], (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0 , 0, 255), thickness=2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(destination_path)
        #cv2.imwrite(filename=destination_path, img=image)

    def roi_pool(self, image: np.array, x1: int, y1: int, x2: int, y2: int):
        """
        ectracts pooled roi of size 3 x self.roi_pool_size x self.roi_pool_size
        :param image:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """
        roi = image[y1:y2, x1:x2, :]  # y is associated to height, x is associated to width
        roi_tensor = self.image_to_tensor(roi)
        roi_tensor = torch.unsqueeze(roi_tensor, dim=0)
        roi_upsampled = self.image_up_or_down_sampler(roi_tensor).squeeze()
        return roi_upsampled
