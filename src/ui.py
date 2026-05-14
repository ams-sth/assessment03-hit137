import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from dataclasses import dataclass

# Import your project modules – adjust paths if needed
from .models import DifferenceRegion
from .generator import DifferenceGenerator

@dataclass
class DisplayImage:
    photo: ImageTk.PhotoImage
    scale: float
    width: int
    height: int

    def to_original(self, x_display: int, y_display: int) -> tuple[int, int]:
        return int(x_display / self.scale), int(y_display / self.scale)


def _bgr_to_photo(image_bgr: np.ndarray, max_side: int = 520) -> DisplayImage:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(w, h)))
    dw, dh = int(w * scale), int(h * scale)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if scale != 1.0:
        pil = pil.resize((dw, dh), Image.Resampling.LANCZOS)
    return DisplayImage(photo=ImageTk.PhotoImage(pil), scale=scale, width=dw, height=dh)


class SpotTheDifferenceUI:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("Spot the Difference — HIT137")
        self.root.resizable(False, False)

        self.generator = DifferenceGenerator()

        self.original_bgr: np.ndarray | None = None
        self.modified_bgr: np.ndarray | None = None
        self.regions: list[DifferenceRegion] = []

        self.total_found = 0
        self.mistakes = 0
        self.max_mistakes = 3
        self.guessing_enabled = False

        self._orig_display: DisplayImage | None = None
        self._mod_display: DisplayImage | None = None

        self._build()

    def _build(self) -> None:
        # Top bar
        top = ctk.CTkFrame(self.root, corner_radius=0)
        top.pack(fill="x")

        ctk.CTkButton(
            top, text="Load image…", width=120, command=self.load_image
        ).pack(side="left", padx=(12, 6), pady=10)

        ctk.CTkButton(
            top, text="Reveal", width=90,
            fg_color="gray40", hover_color="gray55",
            command=self.reveal
        ).pack(side="left", padx=(0, 12), pady=10)

        self.status_var = tk.StringVar(value="Load an image to start.")
        ctk.CTkLabel(top, textvariable=self.status_var, anchor="w").pack(
            side="left", padx=(8, 0)
        )

        # Score bar
        score_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="gray17")
        score_frame.pack(fill="x")

        self.score_var = tk.StringVar(value="Remaining: 0    Mistakes: 0/3    Total found: 0")
        ctk.CTkLabel(
            score_frame, textvariable=self.score_var,
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=6, padx=12, anchor="w")

        # Main body frame – now stored as self.body to avoid AttributeError
        self.body = ctk.CTkFrame(self.root, corner_radius=0, fg_color="gray12")
        self.body.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(self.body, corner_radius=8, fg_color="gray20")
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        right = ctk.CTkFrame(self.body, corner_radius=8, fg_color="gray20")
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))

        ctk.CTkLabel(left, text="Original", text_color="gray70").pack(pady=(6, 2))
        ctk.CTkLabel(right, text="Modified  ·  click to guess", text_color="gray70").pack(pady=(6, 2))

        self.canvas_orig = tk.Canvas(left, bg="#1a1a1a", highlightthickness=0)
        self.canvas_orig.pack(padx=8, pady=(0, 8))

        self.canvas_mod = tk.Canvas(right, bg="#1a1a1a", highlightthickness=0)
        self.canvas_mod.pack(padx=8, pady=(0, 8))

        self.canvas_mod.bind("<Button-1>", self.on_click_modified)

    def _update_score_labels(self) -> None:
        remaining = sum(1 for r in self.regions if not r.found)
        self.score_var.set(
            f"Remaining: {remaining}    Mistakes: {self.mistakes}/{self.max_mistakes}    Total found: {self.total_found}"
        )

    def _clear_canvases(self) -> None:
        self.canvas_orig.delete("all")
        self.canvas_mod.delete("all")

    def _render_images(self) -> None:
        if self.original_bgr is None or self.modified_bgr is None:
            return
        self._orig_display = _bgr_to_photo(self.original_bgr)
        self._mod_display = _bgr_to_photo(self.modified_bgr)

        self._clear_canvases()
        for canvas, display in (
            (self.canvas_orig, self._orig_display),
            (self.canvas_mod, self._mod_display),
        ):
            canvas.config(width=display.width, height=display.height)
            canvas.create_image(0, 0, image=display.photo, anchor="nw", tags=("img",))

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        img = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Load failed", "Could not read this image.")
            return

        try:
            modified, regions = self.generator.generate(img)
        except Exception as e:
            messagebox.showerror("Generation failed", str(e))
            return

        self.original_bgr = img
        self.modified_bgr = modified
        self.regions = regions
        self.mistakes = 0
        self.total_found = 0          # reset total found
        self.guessing_enabled = True  # re-enable guessing

        self.status_var.set("Find all 5 differences by clicking the modified image.")
        self._render_images()
        self._update_score_labels()
    
    def _draw_circle_on_both(self, region: DifferenceRegion, color: str) -> None:
        if self._orig_display is None or self._mod_display is None:
            return
        s = self._orig_display.scale
        cx = int(region.cx * s)
        cy = int(region.cy * s)
        r = int(region.radius * s)
        for canvas in (self.canvas_orig, self.canvas_mod):
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=3)

    def on_click_modified(self, event: tk.Event) -> None:
        if not self.guessing_enabled or self._mod_display is None:
            return

        ox, oy = self._mod_display.to_original(event.x, event.y)

        hit = next(
            (
                r for r in self.regions
                if not r.found
                and ((ox - r.cx) ** 2 + (oy - r.cy) ** 2) ** 0.5 <= r.radius
            ),
            None,
        )

        if hit is None:
            self.mistakes += 1
            self._update_score_labels()

            if self.mistakes >= self.max_mistakes:
                # Reveal remaining differences and show game over popup
                self._reveal_remaining(show_popup=True)
            else:
                self.status_var.set(f"Not quite — {self.max_mistakes - self.mistakes} guess(es) left.")
            return

        # Correct guess
        hit.found = True
        self.total_found += 1
        self._draw_circle_on_both(hit, color="#ff4444")

        remaining = sum(1 for r in self.regions if not r.found)
        if remaining == 0:
            self._end_game(won=True)   # keep your existing win method
        else:
            self.status_var.set(f"Found one! {remaining} to go.")
        self._update_score_labels()
    
    def reveal(self) -> None:
        """Reveal all unfound differences with blue circles and disable guessing."""
        self._reveal_remaining(show_popup=False)
    
    def _end_game(self, won: bool) -> None:
        """Disable guessing and show a final message."""
        self.guessing_enabled = False
        if won:
            messagebox.showinfo("🎉 You won!", f"You found all {len(self.regions)} differences!")
            self.status_var.set(f"You won! Found all {len(self.regions)}. Load a new image to play again.")
        else:
            messagebox.showwarning("💀 Game Over", f"Too many mistakes ({self.max_mistakes}). Better luck next time!")
            self.status_var.set(f"Game over! You made {self.max_mistakes} mistakes. Load a new image to try again.")
    
    def _reveal_remaining(self, show_popup: bool = False) -> None:
        """Draw blue circles on all unfound differences and disable guessing."""
        if not self.regions:
            return

        for r in self.regions:
            if not r.found:
                self._draw_circle_on_both(r, color="#4499ff")

        self.guessing_enabled = False

        if show_popup:
            remaining = sum(1 for r in self.regions if not r.found)
            messagebox.showinfo(
                "Game Over",
                f"You've used all {self.max_mistakes} mistakes.\n"
                f"{remaining} difference(s) remaining have been revealed."
            )
            self.status_var.set(f"Game over – revealed {remaining} remaining differences. Load a new image.")
        else:
            self.status_var.set("Revealed. Load a new image to restart.")