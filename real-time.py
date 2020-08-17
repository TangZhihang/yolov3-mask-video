from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
import os
import time
import datetime
import argparse

from PIL import Image
import torch
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt

# --rtsp "rtsp://admin:zjlab123@10.0.105.148:554/h264/ch1/main/av_stream"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", type=str, default="rtsp://admin:zjlab123@10.0.105.148:554/h264/ch1/main/av_stream", help="url to video stream")
    parser.add_argument("--model_def", type=str, default="config/yolov3-mask.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_face_mask.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="config/face_mask_data.names",
                        help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    if opt.rtsp:
        capture = cv2.VideoCapture(opt.rtsp)
    else:
        capture = cv2.VideoCapture(0)
    while (True):
        ref, image = capture.read()
        if not ref:  # if not return image
            continue
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        image_pil = np.array(image_pil)

        # Pad to square resolution
        img, _ = pad_to_square(image, 0)
        img = transforms.ToTensor()(img)  # img为np.uint8格式

        # Resize
        img = resize(img, opt.img_size)

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        print("\nPerforming object detection:")
        prev_time = time.time()

        # Configure input
        input_img = torch.Tensor(img).unsqueeze(0)

        input_img = Variable(input_img.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_img)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ , Inference Time: %s" % (inference_time))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            if detections[0] is None:
                continue
            detections = rescale_boxes(detections[0], opt.img_size, image_pil.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            cls2color = {0: (0, 0, 255), 1: (0, 255, 0)}
            unmask_num = 0
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print(cls_pred)
                if int(cls_pred) == 0:
                    unmask_num += 1
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                box_w = x2 - x1
                box_h = y2 - y1
                cv2.rectangle(image, (x1, y1), (x1 + box_w, y1 + box_h), cls2color[int(cls_pred)], 2)
                text = "{}: {:.4f}".format(classes[int(cls_pred)], cls_conf.item())
                cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("ZhejiangLab001", image)
            c = cv2.waitKey(30) & 0xff
            if c == 27:
                capture.release()
                break
            # Save generated image with detections
            if unmask_num:
                filename = timestamp
                string = timestamp + ' ' + "unmask boxes num:"
                cv2.imwrite(f"output/{filename}.png", image)
