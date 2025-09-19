import os
import cv2
import math
import json
import time
import random
import numpy as np
import torch
from ultralytics import YOLO

from PIL import Image
from typing import List, Tuple, Dict, Optional

VIDEO_PATH = "//mnt/c/Users/rdulam/Downloads/BCS_sample_video.mp4"
OUT_DIR = "//mnt/c/Users/rdulam/Downloads/Precision Livestock Farming/BCS_Results/"
model_name = "yolov9e-seg-bytetrack_custom_v5_long"
DETECT_EVERY_N_FRAMES = 1
OUT_VIDEO_PATH = os.path.join(OUT_DIR, "cows_segmented_" + model_name + "_" + str(DETECT_EVERY_N_FRAMES) + ".mp4")
SAVE_MASKS = True
BOX_THRESH = 0.30
TEXT_THRESH = 0.25
TEXT_PROMPT = "cow"
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# --------------- Utilities ---------------
def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
	return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
	inter_x0 = max(a[0], b[0]); inter_y0 = max(a[1], b[1])
	inter_x1 = min(a[2], b[2]); inter_y1 = min(a[3], b[3])
	iw = max(0, inter_x1 - inter_x0); ih = max(0, inter_y1 - inter_y0)
	inter = iw * ih
	area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
	area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
	union = area_a + area_b - inter + 1e-6
	return inter / union

def colored_mask(mask: np.ndarray, color: Tuple[int,int,int], alpha: float=0.45) -> np.ndarray:
	"""Return an RGBA overlay for a HxW boolean mask."""
	h, w = mask.shape
	overlay = np.zeros((h, w, 3), dtype=np.uint8)
	overlay[mask] = color
	return overlay, alpha

def ensure_uint8(img):
	if img.dtype != np.uint8:
		img = np.clip(img, 0, 255).astype(np.uint8)
	return img

def assign_color_for_id(obj_id: int) -> Tuple[int,int,int]:
	random.seed(obj_id + 12345)
	return (random.randint(64,255), random.randint(64,255), random.randint(64,255))

def write_mask_png(mask: np.ndarray, path: str):
	cv2.imwrite(path, (mask.astype(np.uint8) * 255))

if __name__ == "__main__":

	# MODEL_NAME = "yolov8n-seg.pt"
	# MODEL_NAME = "yolov8s-seg.pt"
	# MODEL_NAME = "yolo11x-seg.pt" 
	MODEL_NAME = "yolov9e-seg.pt"
	CONF = 0.25
	IOU = 0.5
	TRACKER_CFG = "bytetrack_custom.yaml"
	TARGET_CLASS = "cow"

	os.makedirs(OUT_DIR, exist_ok=True)
	if SAVE_MASKS:
		os.makedirs(os.path.join(OUT_DIR, "masks_" + model_name + "_" + str(DETECT_EVERY_N_FRAMES)), exist_ok=True)
	
    model = YOLO(MODEL_NAME)

	t0 = time.time()
	names = model.model.names if hasattr(model, "model") else model.names
	if isinstance(names, dict):
		cow_ids = [i for i, n in names.items() if str(n).lower() == TARGET_CLASS]
	else:
		cow_ids = [i for i, n in enumerate(names) if str(n).lower() == TARGET_CLASS]
	assert len(cow_ids) >= 1, "Model does not have a 'cow' class."
	COW_ID = cow_ids[0]

	cap = cv2.VideoCapture(VIDEO_PATH)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open {VIDEO_PATH}")
	W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, FPS, (W, H))

	results = model.track(
		source=VIDEO_PATH,
		conf=CONF,
		iou=IOU,
		tracker=TRACKER_CFG,
		stream=True,          
		classes=[COW_ID],     
		persist=True,    
		verbose=False,
		vid_stride=DETECT_EVERY_N_FRAMES
	)

	frame_idx = -1
	for r in results:
		frame_idx += 1
		frame = r.orig_img.copy()
		if r.masks is None or r.boxes is None:
			writer.write(frame)
			continue

		boxes = r.boxes.xyxy.cpu().numpy()
		ids = (r.boxes.id.cpu().numpy().astype(int)
			if r.boxes.id is not None else np.arange(len(boxes)))
		clses = r.boxes.cls.cpu().numpy().astype(int)
		masks = r.masks.data.cpu().numpy()
		overlay = frame.copy()
		for k in range(len(boxes)):
			if clses[k] != COW_ID:
				continue
			mask = masks[k] > 0.5
			mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
			color = (int(80 + (ids[k]*53) % 175),
					int(80 + (ids[k]*97) % 175),
					int(80 + (ids[k]*131) % 175))
			overlay[mask] = (0.55*overlay[mask] + 0.45*np.array(color)).astype(np.uint8)
			cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(overlay, cnts, -1, color, 2)
			ys, xs = np.where(mask)
			if len(xs):
				cx, cy = int(xs.mean()), int(ys.mean())
				cv2.putText(overlay, f"id {ids[k]}",
							(cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

			if SAVE_MASKS:
				out_mask = (mask.astype(np.uint8) * 255)
				cv2.imwrite(os.path.join(OUT_DIR, "masks_" + model_name + "_" + str(DETECT_EVERY_N_FRAMES), f"frame_{frame_idx:06d}_id_{ids[k]}.png"), out_mask)

		frame_vis = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0.0)
		writer.write(frame_vis)

	cap.release()
	writer.release()
	print(f"Done. Video: {OUT_VIDEO_PATH}  |  Time: {time.time()-t0:.1f}s")
