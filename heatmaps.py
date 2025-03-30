import os
import shutil
import cv2
import torch
import numpy as np
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import RTDETRDetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return logits_[indices], boxes_[indices], xywh2xyxy(boxes_[indices]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        h, w, _ = img.shape
        box[0] = box[0] * w
        box[2] = box[2] * w
        box[1] = box[1] * h
        box[3] = box[3] * h
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_folder_path, save_path):
        # Create result directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Loop through all images in the folder
        for img_name in os.listdir(img_folder_path):
            img_path = os.path.join(img_folder_path, img_name)

            if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Create a subdirectory for each image's results
                img_result_path = os.path.join(save_path, img_name.split('.')[0])
                os.makedirs(img_result_path, exist_ok=True)

                # Process the image
                print(f"Processing {img_name}...")
                img = cv2.imread(img_path)
                img = letterbox(img, auto=False, scaleFill=True)[0]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.float32(img) / 255.0
                tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

                # Initialize ActivationsAndGradients
                grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

                # Get activations and results
                result = grads(tensor)
                activations = grads.activations[0].cpu().detach().numpy()

                # Post-process to YOLO output
                post_result, pre_post_boxes, post_boxes = self.post_process(result[0][0])

                for i in trange(int(post_result.size(0) * self.ratio)):
                    if float(post_result[i].max()) < self.conf_threshold:
                        break

                    self.model.zero_grad()

                    # Get max probability for this prediction
                    if self.backward_type == 'class' or self.backward_type == 'all':
                        score = post_result[i].max()
                        score.backward(retain_graph=True)

                    if self.backward_type == 'box' or self.backward_type == 'all':
                        for j in range(4):
                            score = pre_post_boxes[i, j]
                            score.backward(retain_graph=True)

                    # Process heatmap
                    if self.backward_type == 'class':
                        gradients = grads.gradients[0]
                    elif self.backward_type == 'box':
                        gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
                    else:
                        gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                                    grads.gradients[4]
                    b, k, u, v = gradients.size()
                    weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                          gradients.detach().numpy())
                    weights = weights.reshape((b, k, 1, 1))
                    saliency_map = np.sum(weights * activations, axis=1)
                    saliency_map = np.squeeze(np.maximum(saliency_map, 0))
                    saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
                    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                    if (saliency_map_max - saliency_map_min) == 0:
                        continue
                    saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

                    # Add heatmap and box to image
                    cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)

                    cam_image = Image.fromarray(cam_image)
                    cam_image.save(f'{img_result_path}/{i}.png')


def get_params():
    params = {
        'weight': '',
        'cfg': '',
        'device': 'cuda:0',
        'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[-2]',
        'backward_type': 'box',  # class, box, all
        'conf_threshold': 0.5,  # 0.3
        'ratio': 0.02  # 0.5-1.0
    }
    return params


if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    img_folder_path = r'F:\visdrone\VisDrone2019-DET-val\images'  # Path to your folder containing images
    result_folder_path = 'result'  # Path to the result folder
    model(img_folder_path, result_folder_path)
