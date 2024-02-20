from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import cv2
from open_clip import image_transform
from ultralytics.data.augment import LetterBox
from ultralytics.data.loaders import LoadPilAndNumpy
from ultralytics.engine.results import Results
# from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics.utils import ops

from configs.config import HashingConfig

config = HashingConfig(yaml_path="configs/256bit.yaml")


def default_image_loader(path):
    img = Image.open(path).convert('RGB')  # RGB, BCHW # torchvision.datapoints.
    return img


def postprocess_segmentation(preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=False):
    """
    Copied from FastSAM code:
    https://github.com/CASIA-IVA-Lab/FastSAM/blob/4d153e909f0ad9c8ecd7632566e5a24e21cf0071/fastsam/predict.py#L14
    """

    p = ops.non_max_suppression(preds[0],
                                conf,
                                iou,
                                agnostic_nms,
                                max_det=100,
                                nc=1)

    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        img_path = "ok"
        if not len(pred):  # save empty boxes
            results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
            continue
        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(orig_img=orig_img, path=img_path, names={}, boxes=pred[:, :6], masks=masks))
    return results


def preprocess_segmentation(img_origin, imgsz=1024):
    """
    FastSAM preprocessing
    """
    h, w = img_origin.shape[:2]
    if h > w:
        scale = min(imgsz / h, imgsz / w)
        inp = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        nw = int(w * scale)
        nh = int(h * scale)
        a = int((nh - nw) / 2)
        inp[: nh, a:a + nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    else:
        scale = min(imgsz / h, imgsz / w)
        inp = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        nw = int(w * scale)
        nh = int(h * scale)
        a = int((nw - nh) / 2)

        inp[a: a + nh, :nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    rgb = np.array([inp], dtype=np.float32) / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))


def yolo_transform(im, input_size, is_mask=False):
    """
    Preprocess images or mask for FastSAM (YOLO)
    """

    def pre_transform(im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes
        # stride: model.predictor.model.stride
        if is_mask:
            im = [x * 255 for x in im]
        return [LetterBox(input_size, auto=auto, stride=32)(image=x) for x in im]

    im = LoadPilAndNumpy(im).im0  # , imgsz=input_size
    im = pre_transform(im)
    im = np.stack(im)

    if not is_mask:
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    im = np.ascontiguousarray(im)  # contiguous

    if is_mask:
        img = np.ones_like(im, dtype=np.uint8)  # torch.ones_like(im, dtype=torch.uint8)
        img[im < 115] = 0
    else:
        img = im.astype(np.float32)  # im.float()
        img /= 255  # 0 - 255 to 0.0 - 1.0
    return img.squeeze(0)


def box2mask(image, box):
    w, h = image.size
    bbox_mask = np.zeros((h, w))
    x1, y1, x2, y2 = map(round, box)
    bbox_mask[y1:y2, x1:x2] = 1
    return bbox_mask


def image_to_np_ndarray(image):
    # From: https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py
    if type(image) is str:
        return np.array(Image.open(image))
    elif issubclass(type(image), Image.Image):
        return np.array(image)
    elif type(image) is np.ndarray:
        return image
    return None


class FastSAMPrompt:
    # For Box and Point Prompt
    # From: https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py
    def __init__(self, image, results, device='cuda'):
        if isinstance(image, str) or isinstance(image, Image.Image):
            image = image_to_np_ndarray(image)
        self.device = device
        self.results = results
        self.img = image

    def _segment_image(self, image, bbox):
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        segmented_image_array = np.zeros_like(image_array)
        x1, y1, x2, y2 = bbox
        segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new('RGB', image.size, (255, 255, 255))
        # transparency_mask = np.zeros_like((), dtype=np.uint8)
        transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        transparency_mask[y1:y2, x1:x2] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image

    def _format_results(self, result, filter=0):
        annotations = []
        n = len(result.masks.data)
        for i in range(n):
            annotation = {}
            mask = result.masks.data[i] == 1.0

            if torch.sum(mask) < filter:
                continue
            annotation['id'] = i
            annotation['segmentation'] = mask.cpu().numpy()
            annotation['bbox'] = result.boxes.data[i]
            annotation['score'] = result.boxes.conf[i]
            annotation['area'] = annotation['segmentation'].sum()
            annotations.append(annotation)
        return annotations

    def filter_masks(annotations):  # filte the overlap mask
        annotations.sort(key=lambda x: x['area'], reverse=True)
        to_remove = set()
        for i in range(0, len(annotations)):
            a = annotations[i]
            for j in range(i + 1, len(annotations)):
                b = annotations[j]
                if i != j and j not in to_remove:
                    # check if
                    if b['area'] < a['area']:
                        if (a['segmentation'] & b['segmentation']).sum() / b['segmentation'].sum() > 0.8:
                            to_remove.add(j)

        return [a for i, a in enumerate(annotations) if i not in to_remove], to_remove

    def _get_bbox_from_mask(self, mask):
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h
        if len(contours) > 1:
            for b in contours:
                x_t, y_t, w_t, h_t = cv2.boundingRect(b)
                # Merge multiple bounding boxes into one.
                x1 = min(x1, x_t)
                y1 = min(y1, y_t)
                x2 = max(x2, x_t + w_t)
                y2 = max(y2, y_t + h_t)
            h = y2 - y1
            w = x2 - x1
        return [x1, y1, x2, y2]

    def _crop_image(self, format_results):

        image = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        ori_w, ori_h = image.size
        annotations = format_results
        mask_h, mask_w = annotations[0]['segmentation'].shape
        if ori_w != mask_w or ori_h != mask_h:
            image = image.resize((mask_w, mask_h))
        cropped_boxes = []
        cropped_images = []
        not_crop = []
        filter_id = []
        # annotations, _ = filter_masks(annotations)
        # filter_id = list(_)
        for _, mask in enumerate(annotations):
            if np.sum(mask['segmentation']) <= 100:
                filter_id.append(_)
                continue
            bbox = self._get_bbox_from_mask(mask['segmentation'])  # mask 的 bbox
            cropped_boxes.append(self._segment_image(image, bbox))
            # cropped_boxes.append(segment_image(image,mask["segmentation"]))
            cropped_images.append(bbox)  # Save the bounding box of the cropped image.

        return cropped_boxes, cropped_images, not_crop, filter_id, annotations

    def box_prompt(self, bbox=None, bboxes=None):
        if self.results == None:
            return []
        assert bbox or bboxes
        if bboxes is None:
            bboxes = [bbox]
        max_iou_index = []
        for bbox in bboxes:
            assert (bbox[2] != 0 and bbox[3] != 0)
            masks = self.results[0].masks.data
            target_height = self.img.shape[0]
            target_width = self.img.shape[1]
            h = masks.shape[1]
            w = masks.shape[2]
            if h != target_height or w != target_width:
                bbox = [
                    int(bbox[0] * w / target_width),
                    int(bbox[1] * h / target_height),
                    int(bbox[2] * w / target_width),
                    int(bbox[3] * h / target_height), ]
            bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
            bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
            bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
            bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

            # IoUs = torch.zeros(len(masks), dtype=torch.float32)
            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

            masks_area = torch.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], dim=(1, 2))
            orig_masks_area = torch.sum(masks, dim=(1, 2))

            union = bbox_area + orig_masks_area - masks_area
            IoUs = masks_area / union
            max_iou_index.append(int(torch.argmax(IoUs)))
        max_iou_index = list(set(max_iou_index))
        return np.array(masks[max_iou_index].cpu().numpy())

    def point_prompt(self, points, pointlabel):  # numpy
        if self.results == None:
            return []
        masks = self._format_results(self.results[0], 0)
        target_height = self.img.shape[0]
        target_width = self.img.shape[1]
        h = masks[0]['segmentation'].shape[0]
        w = masks[0]['segmentation'].shape[1]
        if h != target_height or w != target_width:
            points = [[int(point[0] * w / target_width), int(point[1] * h / target_height)] for point in points]
        onemask = np.zeros((h, w))
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        for i, annotation in enumerate(masks):
            if type(annotation) == dict:
                mask = annotation['segmentation']
            else:
                mask = annotation
            for i, point in enumerate(points):
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                    onemask[mask] = 1
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                    onemask[mask] = 0
        onemask = onemask >= 1
        return np.array([onemask])

    def everything_prompt(self):
        if self.results == None:
            return []
        return self.results[0].masks.data


def get_context_box(image, box, small_box_area, medium_box_area, context_crop_factor_small,
                    context_crop_factor_medium) -> Tuple:
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    image_w, image_h = image.size

    # Resize crop box
    box_x_center = x1 + (x2 - x1) / 2
    box_y_center = y1 + (y2 - y1) / 2
    if box_w * box_h <= small_box_area:
        scale_factor = context_crop_factor_small
    elif box_w * box_h <= medium_box_area:
        scale_factor = context_crop_factor_medium
    x1_new = max(0, box_x_center - 0.5 * box_w * scale_factor)
    x2_new = min(image_w, box_x_center + 0.5 * box_w * scale_factor)
    y1_new = max(0, box_y_center - 0.5 * box_h * scale_factor)
    y2_new = min(image_h, box_y_center + 0.5 * box_h * scale_factor)
    box = x1_new, y1_new, x2_new, y2_new
    return box


def resize2context(original_image, boxes, context, small_box_area, medium_box_area, context_crop_factor_small,
                   context_crop_factor_medium):
    context_boxes = []
    for box in boxes:
        image = original_image.copy()
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        if (context == "small" and box_w * box_h <= small_box_area) or (
                context == "medium" and box_w * box_h <= medium_box_area):
            context_box = get_context_box(image, box, small_box_area, medium_box_area, context_crop_factor_small,
                                          context_crop_factor_medium)
        else:
            context_box = box
        context_boxes.append(context_box)

    return context_boxes


def preprocess_hash(segmentation_results, image_path: str, box, device,
                    fastsam_input_size=1024, clip_input_size=224):
    image = default_image_loader(image_path)
    fastsam_input_image = image.copy()

    prompt_process = FastSAMPrompt(fastsam_input_image, segmentation_results, device=device)

    image_w, image_h = image.size

    pred_mask = prompt_process.box_prompt(bbox=box)
    # Ultralytics format differs from FastSAM implementation version
    # pred_mask = prompt_process.box_prompt(bbox=box)[0].masks.data
    # model_params_type = pred_mask #.dtype
    # pred_mask = pred_mask.cpu().numpy()  # ['masks']

    if len(pred_mask) < 1:
        print("No objects found, using box as mask")
        pred_mask = box2mask(image, box)
    else:

        try:
            pred_mask = np.squeeze(pred_mask).astype(np.uint8)
            pred_mask = cv2.resize(pred_mask, dsize=(image_w, image_h), interpolation=cv2.INTER_NEAREST)
        except:
            print("Image Box (%d,%d)(%d,%d) too small, using box as mask" % (*box,))
            pred_mask = box2mask(image, box)
    mask = pred_mask

    # Use context if object is too small

    if config.features.context_clip != 'none':
        clip_box = resize2context(image, [box], config.features.context_clip, config.features.small_box_area,
                                  config.features.medium_box_area,
                                  config.features.context_crop_factor_small, config.features.context_crop_factor_medium
                                  )[0]
    else:
        clip_box = box

    # clip_crops = []
    # fastsam_crops = []
    # mask_crops = []

    cropped_image = image.crop(box)

    x1, y1, x2, y2 = map(round, box)
    cropped_mask = mask.copy()
    cropped_mask = cropped_mask[y1:y2, x1:x2]
    fastsam_crop = yolo_transform([cropped_image], config.model.fastsam_input_size)
    mask_crop = yolo_transform([np.expand_dims(cropped_mask, 2)], config.model.fastsam_input_size,
                               is_mask=True)  # .unsqueeze(0)

    #
    #
    # x1, y1, x2, y2 = map(round, box) # int
    # mask_crops = yolo_transform([mask[y1:y2, x1:x2]], fastsam_input_size,
    #                             is_mask=True)  # .unsqueeze(0) #.type_as(model_params_type).to(device)
    # fastsam_crops = yolo_transform([cropped_image], fastsam_input_size)  # .type_as(model_params_type).to(device)
    #
    #
    # clip_crops = open_clip_image_transform(cropped_image, is_train=False, resize_longest_max=True,  image_size=clip_input_size)  # .type_as(model_params_type).to(device)

    # Masking
    x1, y1, x2, y2 = map(round, clip_box)  # TODO
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w * box_h > config.features.medium_box_area:
        # x1, y1, x2, y2 = map(round, box)
        cropped_mask = mask[y1:y2, x1:x2]
        cropped_image = Image.fromarray(cropped_image * cropped_mask[..., np.newaxis].astype(np.uint8))

    clip_crop = open_clip_image_transform(cropped_image, is_train=False, resize_longest_max=True,
                                          image_size=config.model.clip_input_size) #.squeeze(0)

    return mask_crop[np.newaxis, np.newaxis, ...].astype(np.float32), fastsam_crop[np.newaxis, ...], clip_crop[
        np.newaxis, ...]


def open_clip_image_transform(image, image_size, is_train=False, resize_longest_max=True, fill_color=0):
    img = image_transform(is_train=is_train, resize_mode="longest",  # resize_longest_max=resize_longest_max,
                          image_size=image_size)(image)  # .squeeze(0)

    # Workaround for padding error in open_clip, see https://github.com/mlfoundations/open_clip/issues/629
    _, height, width = img.shape

    scale = image_size / float(max(height, width))

    if width != height:
        new_size = tuple(round(dim * scale) for dim in (height, width))
        pad_h = image_size - new_size[0]
        pad_w = image_size - new_size[1]
        img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=fill_color)
    return img.numpy()
