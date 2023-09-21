import cv2
import numpy as np
import torch
import time
from segment_anything import SamPredictor, sam_model_registry
from easydict import EasyDict as edict

from raft.raft import RAFT
from utils.utils import InputPadder, \
                        mask2point, mask_spread, point2mask, mask_AABB

global rect_start, rect_end, rect_temp

def on_mouse(event, x, y, flags, param):
    global rect_start, rect_end, rect_temp
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = np.array([x, y])
    elif event == cv2.EVENT_MOUSEMOVE:
        rect_temp = np.array([x, y])
    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = np.array([x, y])

def main():
    global rect_start, rect_end, rect_temp
    rect_start, rect_end, rect_temp = None, None, None

    raft_checkpoint = "/home/star/Develop/Flow_Track/models/raft-things.pth"
    sam_checkpoint = "/home/star/Develop/Flow_Track/models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    video_path = "/home/star/Downloads/VID_20230920_153508.mp4"
    device = "cuda"
    downsample = 0.5
    mask_trans = 0.2
    mask_color = np.array([0.0, 255.0, 0.0])
    cv2.namedWindow('Video Frame', cv2.WINDOW_AUTOSIZE)
    mask_expansion = 0.1

    args = edict({"small": False,
                  "mixed_precision": False,
                  "alternate_corr": False})

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    predictor = SamPredictor(sam)

    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(raft_checkpoint))
    raft_model = raft_model.module
    raft_model.to(device)
    raft_model.eval()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (0, 0), fx=downsample, fy=downsample)

        if frame_count == 0:
            padder = InputPadder(frame.shape[:2])
            cv2.setMouseCallback('Video Frame', on_mouse)
            while True:
                frame_show = frame.copy()
                if rect_start is not None:
                    cv2.rectangle(frame_show, 
                                (rect_start[0], rect_start[1]), 
                                (rect_temp[0], rect_temp[1]), (255, 0, 0), 2, 8, 0)
                
                cv2.imshow('Video Frame', frame_show)
                
                if rect_end is not None:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            if rect_start[0] > rect_end[0]:
                rect_start[0], rect_end[0] = rect_end[0], rect_start[0]
            if rect_start[1] > rect_end[1]:
                rect_start[1], rect_end[1] = rect_end[1], rect_start[1]
            rectangle = np.concatenate([rect_start, rect_end])
            
        else:
            last_frame_torch = torch.from_numpy(last_frame) \
                                .permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            frame_torch = torch.from_numpy(frame) \
                                .permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            
            last_frame_torch, frame_torch = padder.pad(last_frame_torch, frame_torch)
            flow_low, flow_up = raft_model(last_frame_torch, frame_torch, iters=20, test_mode=True)
            
            mask_torch = torch.from_numpy(mask).to(device)
            point, mask_point = mask2point(mask_torch)
            spread_point = mask_spread(mask_torch, mask_point, flow_up[0])
            new_mask = point2mask(mask_torch, spread_point)
            
            rectangle = mask_AABB(spread_point).cpu().numpy()
            rec_expansion =  np.array([rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]]) * mask_expansion
            rectangle = rectangle + np.array([-rec_expansion[0], -rec_expansion[0], 
                                               rec_expansion[1], rec_expansion[1]])
            if rectangle[0] < 0:
                rectangle[0] = 0
            if rectangle[1] < 0:
                rectangle[1] = 0
            if rectangle[2] > frame.shape[1]:
                rectangle[2] = frame.shape[1]
            if rectangle[3] > frame.shape[0]:
                rectangle[3] = frame.shape[0]
            
        input_box = rectangle
        predictor.set_image(frame)
        if frame_count == 0:
            masks, _, _ = predictor.predict(point_coords = None,
                                            point_labels = None,
                                            box = input_box,
                                            multimask_output = False)
        else:
            mid_point = np.array([[(rectangle[0] + rectangle[2]) / 2, (rectangle[1] + rectangle[3]) / 2]])
            masks, _, _ = predictor.predict(point_coords = mid_point.astype(np.int32),
                                            point_labels = np.array([0]),
                                            box = input_box,
                                            multimask_output = False)
            
        mask = masks[0]
        frame_count += 1
        last_frame = frame
        
        frame_show = frame.copy()
        frame_show = frame_show.astype(np.float32)
        frame_show[mask] = (1 - mask_trans) * frame_show[mask] + mask_trans * mask_color
        frame_show = frame_show.astype(np.uint8)
        
        cv2.rectangle(frame_show, rectangle[:2].astype(np.int32), rectangle[-2:].astype(np.int32), 
                      (0, 0, 255), 2, 8, 0)
        
        cv2.imshow('Video Frame', frame_show)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
