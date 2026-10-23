# Yoga Pose Checker

Upload a photo (or use the webcam) and get a skeleton overlay, joint angles and
plain-English corrections for 15 yoga poses. Built on MediaPipe Pose (33 landmarks).

## Setup (Windows, Python 3.12)

```powershell
cd C:\Yoga
py -3.12 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

MediaPipe 0.10.x needs `numpy<2`, so keep the project in its own virtual environment.

## Run

```powershell
venv\Scripts\python -m streamlit run app.py      # web app: upload or take a photo
venv\Scripts\python live.py --pose tree          # webcam mode, press q to quit
venv\Scripts\python live.py --list               # all poses and their short names
venv\Scripts\python check_images.py test_images  # batch-test a folder of photos
```

`check_images.py` prints the detected pose, score and feedback for every image and
writes annotated copies to `test_images/out/`. If a filename starts with a pose short
name (`tree_1.jpg`, `cobra_2.jpg`, ...) it also reports whether auto-detect agreed.

## Poses

| Pose | Short names | Shoot from |
|---|---|---|
| Mountain Pose | mountain, tadasana | front |
| Tree Pose | tree, vrikshasana | front |
| Chair Pose | chair, utkatasana | front |
| Goddess Pose | goddess | front |
| Warrior I | warrior1 | front |
| Warrior II | warrior2, warrior | front |
| Triangle Pose | triangle, trikonasana | front |
| Low Lunge | lunge, anjaneyasana | side |
| Downward Dog | downdog | side |
| Upward Dog | updog | side |
| Cobra Pose | cobra, bhujangasana | side |
| Plank | plank | side |
| Bridge Pose | bridge | side |
| Boat Pose | boat, navasana | side |
| Child's Pose | child, balasana | side |

Target angles and tolerances live in `POSES` in [pose_utils.py](pose_utils.py).
Each joint angle is measured between three landmarks in the 2D image
(elbow: shoulder-elbow-wrist, shoulder: elbow-shoulder-hip, hip: shoulder-hip-knee,
knee: hip-knee-ankle). The values are sensible starting points, not teaching
standards; tune them to your own reference photos.

## How scoring works

* A joint scores 100 if it is within tolerance of the target, then drops linearly
  to 0 over the next 60 degrees.
* The pose score is the mean over joints that MediaPipe saw clearly
  (visibility >= 0.5). Poses marked `mirror` are tried on both sides.
* Auto-detect ranks all 15 poses and flags the result as low-confidence when fewer
  than half the joints were visible, the runner-up is within 5 points, or the best
  score is under 60.

## Test images

`test_images/` holds Creative Commons photos from Wikimedia Commons, two per pose,
named `<shortname>_<n>.jpg`. Sources and licences are listed in
`test_images/CREDITS.txt`. They are for local testing only; check each licence
before redistributing.

## Limits

Angles are measured in 2D, so camera angle matters a lot: front-view poses need the
camera in front, floor poses need a side view. Photos with several people, tiny
figures, silhouettes or heavy grain may not be detected.
