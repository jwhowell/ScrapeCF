"""
draw_boxes.py

Interactive CLI tool to draw bounding boxes on an image using OpenCV.

Usage:
    python draw_boxes.py path/to/image.jpg
    python draw_boxes.py path/to/image.jpg --output boxes.json
    python draw_boxes.py path/to/image.jpg --load existing_boxes.json

Controls:
    Left mouse button drag : draw a box
    u                      : undo last box
    d                      : delete box under mouse (press 'd' while mouse over a box)
    r                      : reset (remove all boxes)
    s                      : save boxes to output file (default: <image_name>_boxes.json)
    q or ESC               : quit without saving (if not saved)
    h                      : print help in console
Notes:
    Saved format: JSON
    {
      "image": "path/to/image.jpg",
      "shape": [height, width],
      "boxes": [ [x, y, w, h], ... ]   # coordinates relative to original image
    }
"""

import argparse
import json
import os

import cv2

MAX_DISPLAY_DIM = 1000  # largest dimension for display window


def scale_for_display(img, max_dim=MAX_DISPLAY_DIM):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        disp = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        disp = img.copy()
    return disp, scale


class BoxEditor:
    def __init__(self, image_path, output_path=None, load_path=None):
        self.image_path = image_path
        self.img_orig = cv2.imread(image_path)
        if self.img_orig is None:
            raise FileNotFoundError(f"Could not open image: {image_path}")
        self.h_orig, self.w_orig = self.img_orig.shape[:2]

        self.disp_img, self.scale = scale_for_display(self.img_orig)
        self.h_disp, self.w_disp = self.disp_img.shape[:2]

        self.window_name = "Draw Boxes - press 'h' for help"
        self.drawing = False
        self.start_pt_disp = None
        self.cur_pt_disp = None
        self.boxes = []  # stored in original image coordinates as [x, y, w, h]
        self.output_path = output_path or self.default_output_name()
        if load_path:
            self.load_boxes(load_path)

        # for deletion under mouse
        self.mouse_pos = (0, 0)

    def default_output_name(self):
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        return f"{base}_boxes.json"

    def disp_to_orig(self, pt_disp):
        x_disp, y_disp = int(round(pt_disp[0])), int(round(pt_disp[1]))
        if self.scale == 0:
            return (x_disp, y_disp)
        x = int(round(x_disp / self.scale))
        y = int(round(y_disp / self.scale))
        # clamp
        x = max(0, min(self.w_orig - 1, x))
        y = max(0, min(self.h_orig - 1, y))
        return x, y

    def orig_to_disp(self, box):
        # box: [x, y, w, h] in original coords -> returns ints in display coords
        x, y, w, h = box
        xd = int(round(x * self.scale))
        yd = int(round(y * self.scale))
        wd = int(round(w * self.scale))
        hd = int(round(h * self.scale))
        return xd, yd, wd, hd

    def add_box_from_disp(self, start_disp, end_disp):
        x0, y0 = self.disp_to_orig(start_disp)
        x1, y1 = self.disp_to_orig(end_disp)
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w == 0 or h == 0:
            return
        self.boxes.append([x, y, w, h])
        print(f"Added box: {[x, y, w, h]} (original coords)")

    def load_boxes(self, path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if "boxes" in data:
                self.boxes = data["boxes"]
                print(f"Loaded {len(self.boxes)} boxes from {path}")
            else:
                print("No 'boxes' key found in JSON. Skipping load.")
        except Exception as e:
            print(f"Could not load boxes from {path}: {e}")

    def save_boxes(self, path=None):
        path = path or self.output_path
        payload = {
            "image": self.image_path,
            "shape": [self.h_orig, self.w_orig],
            "boxes": self.boxes,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {len(self.boxes)} boxes to {path}")
        self.output_path = path

    def box_under_disp_point(self, pt_disp):
        # returns index of box under display point or None
        for i, box in enumerate(self.boxes):
            xd, yd, wd, hd = self.orig_to_disp(box)
            rx, ry, rw, rh = xd, yd, wd, hd
            x, y = pt_disp
            if x >= rx and x <= rx + rw and y >= ry and y <= ry + rh:
                return i
        return None

    def draw(self):
        # draw overlay on a copy of disp_img
        canvas = self.disp_img.copy()
        # draw existing boxes
        for box in self.boxes:
            xd, yd, wd, hd = self.orig_to_disp(box)
            cv2.rectangle(canvas, (xd, yd), (xd + wd, yd + hd), (0, 255, 0), 2)
        # draw current drawing rect
        if self.drawing and self.start_pt_disp and self.cur_pt_disp:
            cv2.rectangle(canvas, self.start_pt_disp, self.cur_pt_disp, (0, 0, 255), 1)
        # draw mouse crosshair small
        mx, my = self.mouse_pos
        cv2.line(canvas, (mx - 10, my), (mx + 10, my), (200, 200, 200), 1)
        cv2.line(canvas, (mx, my - 10), (mx, my + 10), (200, 200, 200), 1)
        # show help text top-left
        cv2.putText(
            canvas,
            f"Boxes: {len(self.boxes)}  (s: save  u: undo  d: delete under mouse  r: reset  q: quit)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt_disp = (x, y)
            self.cur_pt_disp = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.cur_pt_disp = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_pt_disp:
                self.cur_pt_disp = (x, y)
                self.add_box_from_disp(self.start_pt_disp, self.cur_pt_disp)
            self.drawing = False
            self.start_pt_disp = None
            self.cur_pt_disp = None

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        saved = False
        print(
            "Instructions: draw boxes by clicking and dragging with left mouse button."
        )
        print(
            "Press 'h' for help. Press 's' to save, 'u' to undo, 'd' to delete a box under the mouse, 'r' to reset, 'q' or ESC to quit."
        )
        while True:
            canvas = self.draw()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 0x1B:  # ESC
                print("ESC pressed. Exiting.")
                break
            if key == ord("q"):
                print("q pressed. Exiting.")
                break
            if key == ord("h"):
                print(
                    self.__doc__
                    if self.__doc__
                    else "Help: use mouse and keys as described."
                )
            if key == ord("s"):
                self.save_boxes()
                saved = True
            if key == ord("u"):
                if self.boxes:
                    removed = self.boxes.pop()
                    print("Undone last box:", removed)
                else:
                    print("No boxes to undo.")
            if key == ord("r"):
                confirm = input("Reset all boxes? type 'yes' to confirm: ")
                if confirm.strip().lower() == "yes":
                    self.boxes = []
                    print("All boxes removed.")
                else:
                    print("Reset cancelled.")
            if key == ord("d"):
                idx = self.box_under_disp_point(self.mouse_pos)
                if idx is not None:
                    removed = self.boxes.pop(idx)
                    print(f"Deleted box #{idx}: {removed}")
                else:
                    print("No box under mouse to delete.")
        cv2.destroyAllWindows()
        if not saved:
            print(
                "Boxes not explicitly saved. If you want to save them, call the script with --output or press 's' before quitting."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive bounding box annotator using OpenCV"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--output", "-o", help="Path to save boxes JSON")
    parser.add_argument("--load", "-l", help="Load existing boxes JSON to edit")
    args = parser.parse_args()

    editor = BoxEditor(args.image, output_path=args.output, load_path=args.load)
    editor.run()


if __name__ == "__main__":
    main()
