import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import cv2
import numpy as np


def draw_texts(img, texts, offset_x=10, offset_y=0, font_scale=0.7, thickness=2):
    h, w, c = img.shape
    color = (0, 0, 255)  # black

    texts = [texts] if type(texts) == str else texts

    for i, text in enumerate(texts):
        cv2.putText(img, text, (offset_x, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)


def detection_fasterrcnn(img_path="./samples/00017.png", finetune=False):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    if finetune:
        num_classes = 2  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)
    model.eval()
    print(model)

    # load color image
    img = cv2.imread(img_path)
    img_tr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tr = np.transpose(img_tr, (2, 0, 1)).astype(np.float32) / 255.0
    print(img.shape)

    t = torch.from_numpy(img_tr).to(device)
    t = t.unsqueeze(0)

    with torch.no_grad():
        out = model(t)

    print(out)

    boxes = out[0]["boxes"].data.cpu().numpy()
    scores = out[0]["scores"].data.cpu().numpy()
    labels = out[0]["labels"].data.cpu().tolist()

    category = {0: 'background', 1: 'person', 2: 'traffic light', 3: 'train', 4: 'traffic sign', 5: 'rider',
                6: 'car', 7: 'bike', 8: 'motor', 9: 'truck', 10: 'bus'}

    boxes = boxes[scores >= 0.5].astype(np.int32)
    for i, box in enumerate(boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))
        if labels[i] == 1:
            draw_texts(img, 'person '+str(round(scores[i], 3)), offset_x=box[0], offset_y=box[1])

    cv2.imshow("result", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    detection_fasterrcnn()
