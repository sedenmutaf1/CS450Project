"""
label_videos.py -- Interactive video query labeling tool.

Plays each video in a directory. Type a query, press Save (or Enter)
to append it to workload.csv, then add more queries or move to the next video.

Usage:
    python label_videos.py --dir ../input_videos/
    python label_videos.py --dir ../input_videos/ --workload workload.csv

Keyboard shortcuts:
    Enter          Save current query
    Ctrl + Right   Next video
    Ctrl + Left    Previous video
"""

import argparse
import csv
import os
import time
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk


# ── Colour palette ─────────────────────────────────────────────────────────────
BG       = "#f8f8f8"
PANEL    = "#ffffff"
ACCENT   = "#3b82f6"    # blue
SUCCESS  = "#22c55e"    # green
DANGER   = "#ef4444"    # red
MUTED    = "#6b7280"
TEXT     = "#111827"
CANVAS_W = 720
CANVAS_H = 405          # 16:9


class VideoLabeler:
    def __init__(self, root: tk.Tk, video_paths: list[str], workload_path: str):
        self.root          = root
        self.video_paths   = video_paths
        self.workload_path = workload_path
        self.current_idx   = 0
        self.playing       = False
        self.cap           = None
        self.photo         = None
        self._after_id     = None
        self.frame_delay   = 33    # ms between frames, updated per video

        self.saved = self._load_existing()

        self._build_ui()
        self._load_video(0)
        self._bind_keys()

    # ── Persistence ─────────────────────────────────────────────────────────
    def _load_existing(self) -> dict:
        """Returns {video_name: [query, ...]} from the CSV (if it exists)."""
        result = {}
        if os.path.exists(self.workload_path):
            with open(self.workload_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    v = row.get("video", "").strip()
                    q = row.get("query", "").strip()
                    if v and q:
                        result.setdefault(v, []).append(q)
        return result

    def _append_to_csv(self, vname: str, query: str):
        write_header = not os.path.exists(self.workload_path)
        with open(self.workload_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["video", "query"])
            writer.writerow([vname, query])

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Video Labeling Tool — CS450")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # ── Header bar ─────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=ACCENT, padx=14, pady=8)
        header.pack(fill="x")

        self.progress_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.progress_var,
                 bg=ACCENT, fg="white",
                 font=("Segoe UI", 11, "bold")).pack(side="left")

        self.filename_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.filename_var,
                 bg=ACCENT, fg="#dbeafe",
                 font=("Segoe UI", 10)).pack(side="left", padx=12)

        # ── Video canvas ───────────────────────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg="black")
        canvas_frame.pack()
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_W, height=CANVAS_H,
                                bg="black", highlightthickness=0)
        self.canvas.pack()

        # ── Query input strip ──────────────────────────────────────────────
        q_outer = tk.Frame(self.root, bg=PANEL, padx=12, pady=10)
        q_outer.pack(fill="x")

        tk.Label(q_outer, text="Query", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")

        q_row = tk.Frame(q_outer, bg=PANEL)
        q_row.pack(fill="x", pady=(4, 0))

        self.query_var = tk.StringVar()
        self.query_entry = tk.Entry(q_row, textvariable=self.query_var,
                                    font=("Segoe UI", 11),
                                    relief="solid", bd=1,
                                    bg="white", fg=TEXT)
        self.query_entry.pack(side="left", fill="x", expand=True, ipady=5)

        save_btn = tk.Button(q_row, text="Save  ↵",
                              command=self._save_query,
                              bg=SUCCESS, fg="white",
                              font=("Segoe UI", 10, "bold"),
                              relief="flat", padx=14, pady=5,
                              cursor="hand2", activebackground="#16a34a")
        save_btn.pack(side="left", padx=(8, 0))

        # ── Saved queries list ─────────────────────────────────────────────
        list_outer = tk.Frame(self.root, bg=PANEL, padx=12, pady=6)
        list_outer.pack(fill="both", expand=True)

        tk.Label(list_outer, text="Queries saved for this video",
                 bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")

        self.listbox = tk.Listbox(list_outer,
                                   font=("Segoe UI", 10),
                                   height=5,
                                   bg="#f1f5f9",
                                   fg=TEXT,
                                   selectbackground=ACCENT,
                                   selectforeground="white",
                                   relief="flat", bd=0,
                                   highlightthickness=1,
                                   highlightcolor="#e2e8f0")
        self.listbox.pack(fill="both", expand=True, pady=(4, 0))

        # Delete selected query button
        tk.Button(list_outer, text="Remove selected",
                  command=self._remove_selected,
                  bg="#fee2e2", fg=DANGER,
                  font=("Segoe UI", 8),
                  relief="flat", padx=6, pady=2,
                  cursor="hand2").pack(anchor="e", pady=(4, 0))

        # ── Navigation bar ─────────────────────────────────────────────────
        nav = tk.Frame(self.root, bg=BG, padx=12, pady=10)
        nav.pack(fill="x")

        tk.Button(nav, text="← Previous  (Ctrl+←)",
                  command=self._prev_video,
                  bg="#e5e7eb", fg=TEXT,
                  font=("Segoe UI", 10),
                  relief="flat", padx=12, pady=6,
                  cursor="hand2").pack(side="left")

        tk.Button(nav, text="Skip  →  (Ctrl+→)",
                  command=self._next_video,
                  bg=ACCENT, fg="white",
                  font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=12, pady=6,
                  cursor="hand2").pack(side="left", padx=8)

        tk.Button(nav, text="Finish",
                  command=self._finish,
                  bg=DANGER, fg="white",
                  font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=12, pady=6,
                  cursor="hand2").pack(side="right")

    def _bind_keys(self):
        self.root.bind("<Return>",          lambda _: self._save_query())
        self.root.bind("<Control-Right>",   lambda _: self._next_video())
        self.root.bind("<Control-Left>",    lambda _: self._prev_video())

    # ── Video playback (main-thread polling via after()) ───────────────────────
    def _load_video(self, idx: int):
        # Cancel any pending frame callback
        self.playing = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None

        if idx < 0 or idx >= len(self.video_paths):
            return

        self.current_idx = idx
        vpath = self.video_paths[idx]
        vname = os.path.basename(vpath)

        self.progress_var.set(f"Video {idx + 1} / {len(self.video_paths)}")
        self.filename_var.set(vname)
        self.root.title(f"CS450 Labeler — {vname}")

        self._refresh_list(vname)
        self.query_var.set("")
        self.query_entry.focus_set()

        self.cap = cv2.VideoCapture(vpath)
        if not self.cap.isOpened():
            self.canvas.delete("all")
            self.canvas.create_text(CANVAS_W // 2, CANVAS_H // 2,
                                    text="Could not open video",
                                    fill="white", font=("Segoe UI", 14))
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_delay = max(1, int(1000 / min(fps, 30)))
        self.playing = True
        self._schedule_frame()

    def _schedule_frame(self):
        """Read and display one frame, then schedule the next one."""
        if not self.playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Loop: rewind to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            h, w  = frame.shape[:2]
            scale = min(CANVAS_W / w, CANVAS_H / h)
            nw, nh = int(w * scale), int(h * scale)
            frame  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo  = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.photo = photo     # prevent GC
            x = (CANVAS_W - nw) // 2
            y = (CANVAS_H - nh) // 2
            self.canvas.delete("all")
            self.canvas.create_image(x, y, anchor="nw", image=self.photo)

        self._after_id = self.root.after(self.frame_delay, self._schedule_frame)

    # ── Query actions ────────────────────────────────────────────────────────
    def _save_query(self):
        query = self.query_var.get().strip()
        if not query:
            return
        vname = os.path.basename(self.video_paths[self.current_idx])
        self._append_to_csv(vname, query)
        self.saved.setdefault(vname, []).append(query)
        self._refresh_list(vname)
        self.query_var.set("")
        self.query_entry.focus_set()

    def _remove_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        vname   = os.path.basename(self.video_paths[self.current_idx])
        queries = self.saved.get(vname, [])
        idx     = sel[0]
        if idx < len(queries):
            queries.pop(idx)
            self._refresh_list(vname)
            self._rewrite_csv()

    def _refresh_list(self, vname: str):
        self.listbox.delete(0, "end")
        for q in self.saved.get(vname, []):
            self.listbox.insert("end", f"  {q}")

    def _rewrite_csv(self):
        """Rewrite the entire CSV from the in-memory dict (after a deletion)."""
        rows = [(v, q) for v, qs in self.saved.items() for q in qs]
        with open(self.workload_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "query"])
            writer.writerows(rows)

    # ── Navigation ───────────────────────────────────────────────────────────
    def _next_video(self):
        if self.current_idx + 1 < len(self.video_paths):
            self._load_video(self.current_idx + 1)
        else:
            messagebox.showinfo("Done", "All videos reviewed!\nworkload.csv is ready.")

    def _prev_video(self):
        if self.current_idx > 0:
            self._load_video(self.current_idx - 1)

    def _finish(self):
        self.playing = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CS450 Video Query Labeling Tool")
    parser.add_argument("--dir",      default="../input_videos/",
                        help="Directory containing videos")
    parser.add_argument("--workload", default="workload.csv",
                        help="Output CSV file (default: workload.csv)")
    args = parser.parse_args()

    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if not os.path.isdir(args.dir):
        print(f"ERROR: directory not found: {args.dir}")
        return

    video_paths = sorted([
        os.path.join(args.dir, f)
        for f in os.listdir(args.dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not video_paths:
        print(f"No videos found in {args.dir}")
        return

    print(f"Found {len(video_paths)} video(s). Opening labeler...")

    root = tk.Tk()
    VideoLabeler(root, video_paths, args.workload)
    root.mainloop()


if __name__ == "__main__":
    main()
