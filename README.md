# Object Tracking Experiment Guide

This guide is meant to help you get started with experimenting on **object tracking algorithms** using the HPRC cluster. The idea is to run a baseline tracking code with **YOLOv9-e + ByteTrack** and then compare it with **BoT-SORT**.

---

## 📌 Plan

* Share tracking code to analyze and understand the concept. Do some baseline runs with **YOLOv9-e + ByteTrack**.
* If possible, compare **ByteTrack** and **BoT-SORT** on the same video file.
* Perform **Output and Result Analysis**:

  * How consistent are the IDs across frames?
  * Do you notice identity switches?
  * How stable is tracking when there are multiple objects?
* Look at **Ways to Optimize**:

  * Adjust tracker parameters like confidence threshold or matching strategy.
  * Experiment with different YOLO models (YOLOv8, YOLOv9, segmentation models).

---

## ⚙️ Setup on HPRC

1. **Load Python Module**

   ```bash
   module purge
   module load GCCcore/12.2.0
   module load Python/3.10.8
   ```

2. **Create Virtual Environment**

   ```bash
   mkdir BRD_tracker
   cd BRD_tracker
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Create a `requirements.txt` file:

   ```
   ultralytics
   opencv-python-headless
   numpy
   Pillow
   torch
   torchvision
   ```

   Then install everything:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the Code

Run the baseline tracker:

```bash
python cow_tracking.py --video input.mp4 --tracker bytetrack
```

Run with BoT-SORT:

```bash
python cow_tracking.py --video input.mp4 --tracker botsort
```

The output videos will appear in the `runs/` directory.

---

## 🔎 References

* YOLO by Ultralytics: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* ByteTrack: [https://github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)
* BoT-SORT: [https://github.com/NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)

---

## 📊 Suggested Experiments

* Compare **ByteTrack vs BoT-SORT** tracking on the same video.
* Switch between YOLO model sizes (`yolov8n.pt`, `yolov8s.pt`, `yolov9e.pt`) and see how accuracy and speed change.
* Compare how long the runs take on HPRC vs your local machine.

---

### What to Submit

1. Run baseline tracking with YOLO + ByteTrack.
2. Compare it with BoT-SORT.
3. Share a short summary that includes:

   * What you observed about the tracking quality.
   * A couple of screenshots or short video snippets from the runs.
   * Any quick ideas you have for making it better.
