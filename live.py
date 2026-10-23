"""Optional real-time webcam mode.
Run:  python live.py --pose tree        (or --pose auto)   Press q to quit.
      python live.py --list              shows every pose and its short names
"""
import argparse
import cv2

from pose_utils import AUTO, POSES, PoseEstimator, analyze, draw_overlay, resolve_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="auto",
                    help="'auto' or a pose name / short name, e.g. tree, warrior1, downdog (see --list)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="list supported poses and exit")
    args = ap.parse_args()
    if args.list:
        for name, spec in POSES.items():
            print(f"{name:28s} short names: {', '.join(spec['aliases'])}   (shoot from the {spec['view']})")
        return
    try:
        pose = resolve_pose(args.pose)
    except ValueError as e:
        raise SystemExit(f"{e}\nRun with --list to see the options.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    with PoseEstimator(static=False) as est:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = analyze(rgb, pose, est)
            if result is not None and result["evaluation"] is not None:
                rgb = draw_overlay(rgb, result)
                lines = [f"{result['pose']}: {result['evaluation']['overall']:.0f}%"] + result["feedback"][:3]
            else:
                lines = ["Step back so your full body is visible"]
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for i, t in enumerate(lines):
                cv2.putText(out, t, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, t, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Yoga Pose Checker (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
