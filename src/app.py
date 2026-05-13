from __future__ import annotations

import customtkinter as ctk

from .ui import SpotTheDifferenceUI

def run_app() -> None:
    root = ctk.CTk()
    SpotTheDifferenceUI(root)
    root.mainloop()
