"""Spilvinduet: kamera, vejledning og klikbare knapper."""

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from game import outcome

BG = "#f3f5f4"
WHITE = "#ffffff"
INK = "#172e2a"
MUTED = "#667872"
GREEN = "#16725a"
LINE = "#dce5e0"


class GameWindow:
    def __init__(self, on_action, on_point, camera_index):
        self.root = tk.Tk()
        self.root.title("Kryds & bolle")
        self.root.configure(bg=BG)
        self.root.geometry("1120x790")
        self.root.minsize(980, 740)
        self.root.option_add("*Font", "{Segoe UI} 10")
        self.on_action = on_action
        self.on_point = on_point
        self.image_bounds = None
        self.photo = None
        self.root.protocol("WM_DELETE_WINDOW", lambda: on_action("quit"))

        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=28, pady=(22, 18))
        tk.Label(header, text="×  ○", font=("Segoe UI", 28, "bold"),
                 bg=BG, fg=GREEN).pack(side="left", padx=(0, 16))
        titles = tk.Frame(header, bg=BG)
        titles.pack(side="left")
        self.label(titles, "Kryds & bolle", 24, bold=True, bg=BG).pack(anchor="w")
        self.label(titles, "Et fysisk bræt. En digital modspiller.", 10, MUTED, bg=BG).pack(anchor="w")
        self.connection = self.label(header, "●  Forbinder kamera …", 10, MUTED, bg=BG)
        self.connection.pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=28)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=300)
        body.rowconfigure(0, weight=1)
        camera_card = tk.Frame(body, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        camera_header = tk.Frame(camera_card, bg=WHITE)
        camera_header.pack(fill="x", padx=18, pady=16)
        self.label(camera_header, "Dit spillebræt", 13, bold=True).pack(side="left")
        self.step = self.label(camera_header, "1 / 2 · Markér brættet", 10, GREEN)
        self.step.pack(side="right")
        tools = tk.Frame(camera_card, bg=WHITE)
        tools.pack(fill="x", padx=12, pady=(0, 10))
        self.rotate_button = self.button(tools, "Rotér 90°", "rotate")
        self.rotate_button.pack(side="left", padx=(0, 6))
        self.mirror_button = self.button(tools, "Spejlvend", "mirror")
        self.mirror_button.pack(side="left", padx=(0, 6))
        self.debug_button = self.button(tools, "Debug · live", "debug")
        self.debug_button.pack(side="right")
        self.debug_window = None
        self.debug_photo = None
        self.video = tk.Label(camera_card, bg="#142823", fg="#e2ebe6", bd=0,
                              compound="center", font=("Segoe UI", 16))
        self.video.pack(fill="both", expand=True, padx=12)
        self.video.bind("<Button-1>", self.click_video)
        self.hint = self.label(camera_card, "", 10, MUTED)
        self.hint.pack(fill="x", padx=18, pady=16)

        sidebar = tk.Frame(body, bg=BG, width=300)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)
        self.label(sidebar, "NÆSTE TRIN", 9, MUTED, bold=True, bg=BG).pack(anchor="w")
        self.heading = self.label(sidebar, "Markér brættet", 19, bold=True, bg=BG)
        self.heading.pack(anchor="w", pady=(4, 6))
        self.message = self.label(sidebar, "", 10, MUTED, bg=BG)
        self.message.configure(wraplength=290, justify="left", height=3, anchor="nw")
        self.message.pack(anchor="w", fill="x")
        self.read_button = self.button(sidebar, "Aflæs brættet  →", "read", primary=True)
        self.read_button.pack(fill="x", pady=(12, 12), ipady=2)

        board_card = tk.Frame(sidebar, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        board_card.pack(fill="x")
        self.label(board_card, "Spillets overblik", 11, bold=True).pack(anchor="w", padx=16, pady=(12, 0))
        self.board = tk.Canvas(board_card, width=260, height=138, bg=WHITE, highlightthickness=0)
        self.board.pack(pady=4)
        legend = tk.Frame(board_card, bg=WHITE)
        legend.pack(fill="x", padx=16, pady=(0, 12))
        self.label(legend, "×  Dig", 10, INK).pack(side="left")
        self.label(legend, "○  Computer", 10, GREEN).pack(side="right")

        settings = tk.Frame(sidebar, bg=BG)
        settings.pack(fill="x", pady=(12, 0))
        self.label(settings, "Kamera", 10, bold=True, bg=BG).pack(anchor="w", pady=(0, 6))
        camera_row = tk.Frame(settings, bg=BG)
        camera_row.pack(fill="x")
        self.camera = ttk.Combobox(camera_row, state="readonly", width=14,
                                   values=[f"Kamera {i}" for i in range(10)])
        self.camera.set(f"Kamera {camera_index}")
        self.camera.pack(side="left", fill="x", expand=True, ipady=4)
        self.button(camera_row, "Skift", "camera").pack(side="left", padx=(8, 0))
        self.button(sidebar, "Markér hjørner igen", "calibrate").pack(fill="x", pady=(8, 6))
        self.button(sidebar, "Nyt spil", "reset").pack(fill="x")
        self.label(self.root, "Mellemrum: aflæs     ·     K: markér hjørner     ·     R: nyt spil     ·     0–9: kamera     ·     Esc: luk",
                   9, MUTED, bg=BG).pack(side="bottom", before=body, pady=12)

        self.root.bind("<KeyPress>", self.keypress)

    def open_debug(self):
        self.debug_window = tk.Toplevel(self.root)
        self.debug_window.title("Debug · Det kameraet genkender")
        self.debug_window.geometry("850x740")
        self.debug_window.minsize(600, 500)
        self.debug_window.configure(bg=BG)
        self.debug_window.protocol("WM_DELETE_WINDOW", lambda: self.on_action("debug"))
        controls = tk.Frame(self.debug_window, bg=BG)
        controls.pack(fill="x", padx=18, pady=14)
        self.label(controls, "Genkendelse", 12, bold=True, bg=BG).pack(side="left")
        self.debug_mode = ttk.Combobox(controls, state="readonly", width=24,
                                       values=("Spilbrikker", "Hverdagsting"))
        self.debug_mode.current(0)
        self.debug_mode.pack(side="right")
        self.debug_mode.bind("<<ComboboxSelected>>", lambda event: self.on_action("debug_mode"))
        self.label(self.debug_window, "Liveanalyse ændrer ikke spillets træk. Viser kun klasser, modellen er trænet til.",
                   10, MUTED, bg=BG).pack(padx=18, anchor="w")
        self.debug_info = self.label(self.debug_window, "Starter genkendelse …", 10, MUTED, bg=BG)
        self.debug_info.configure(wraplength=780, justify="left", height=3, anchor="nw")
        self.debug_info.pack(fill="x", padx=18, pady=10)
        self.debug_video = tk.Label(self.debug_window, bg="#142823")
        self.debug_video.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.debug_button.configure(text="Luk debug")

    def close_debug(self):
        self.debug_window.destroy()
        self.debug_window = None
        self.debug_photo = None
        self.debug_button.configure(text="Debug · live")

    def clear_debug(self, message="Afventer nyt billede …"):
        if self.debug_window is not None:
            self.debug_video.configure(image="")
            self.debug_photo = None
            self.debug_info.configure(text=message)

    def show_debug(self, frame, findings, info="Livevideo"):
        if self.debug_window is None:
            return
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((max(1, self.debug_video.winfo_width()),
                         max(1, self.debug_video.winfo_height())), Image.Resampling.BILINEAR)
        self.debug_photo = ImageTk.PhotoImage(image)
        self.debug_video.configure(image=self.debug_photo)
        summary = " · ".join(findings[:3]) if findings else "Ingen aktuelle fund."
        self.debug_info.configure(text=f"{info}\n{summary}")

    @staticmethod
    def label(parent, text, size=10, color=INK, bold=False, bg=WHITE):
        return tk.Label(parent, text=text, bg=bg, fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"))

    def button(self, parent, text, action, primary=False):
        return tk.Button(parent, text=text, command=lambda: self.on_action(action),
                         bg=GREEN if primary else WHITE, fg=WHITE if primary else INK,
                         activebackground="#105c48" if primary else "#e5ece8",
                         activeforeground=WHITE if primary else INK, relief="flat", bd=0,
                         padx=14, pady=7, cursor="hand2", font=("Segoe UI", 11, "bold" if primary else "normal"))

    def keypress(self, event):
        # Lad kameravælgeren beholde sine almindelige tastaturfunktioner.
        if event.widget == self.camera and event.keysym != "Escape":
            return
        action = {"space": "read", "r": "reset", "k": "calibrate",
                  "q": "quit", "Escape": "quit"}.get(event.keysym)
        if event.char in "0123456789" and event.char:
            self.camera.set(f"Kamera {event.char}")
            action = "camera"
        if action:
            self.on_action(action)
            return "break"

    def click_video(self, event):
        if self.image_bounds:
            left, top, width, height, source_width, source_height = self.image_bounds
            if left <= event.x < left + width and top <= event.y < top + height:
                self.on_point((event.x - left) * source_width / width,
                              (event.y - top) * source_height / height)

    def show(self, frame, game, calibrated, points, ready, status, camera_index):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        source_width, source_height = image.size
        width, height = max(1, self.video.winfo_width()), max(1, self.video.winfo_height())
        image.thumbnail((width, height), Image.Resampling.BILINEAR)
        self.image_bounds = ((width - image.width) // 2, (height - image.height) // 2,
                             image.width, image.height, source_width, source_height)
        self.photo = ImageTk.PhotoImage(image)
        self.video.configure(image=self.photo,
                             text="" if ready else "Intet kamerabillede\nVælg dit USB-kamera til højre")
        self.connection.configure(text=f"●  Kamera {camera_index} · {'Tilsluttet' if ready else 'Intet billede'}",
                                  fg=GREEN if ready else "#a35b2b")
        result = outcome(game.board)
        if not ready:
            heading, hint = "Vælg dit kamera", "Tilslut USB-kameraet, vælg et nummer og klik Skift."
        elif not calibrated:
            corners = ("øverste venstre", "øverste højre", "nederste højre", "nederste venstre")
            heading = "Markér brættet"
            hint = f"Klik på {corners[len(points)]} hjørne ({len(points) + 1}/4)." if len(points) < 4 else "Klik Markér hjørner igen for at prøve på ny."
        elif result:
            heading = {"X": "Du vandt!", "O": "Computeren vandt", "draw": "Uafgjort!"}[result]
            hint = "Godt spil. Klik Nyt spil, og fjern alle brikker."
        elif game.pending is not None:
            row, col = divmod(game.pending, 3)
            heading = "Computerens træk"
            hint = f"Læg O i række {row + 1}, kolonne {col + 1}. Bekræft med Aflæs brættet."
        else:
            heading, hint = "Din tur", "Læg én X-brik, fjern hænderne, og klik Aflæs brættet."
        self.heading.configure(text=heading)
        self.message.configure(text=(status or hint) if ready else hint)
        self.hint.configure(text=hint, wraplength=max(200, width - 24), justify="left")
        self.step.configure(text="2 / 2 · Klar til spil" if calibrated else f"1 / 2 · Hjørner {len(points)}/4")
        self.read_button.configure(state="normal" if ready and calibrated and not result else "disabled",
                                   text="Bekræft O-brikken  →" if game.pending is not None else "Aflæs brættet  →")
        self.draw_overview(game)

    def draw_overview(self, game):
        self.board.delete("all")
        for i, piece in enumerate(game.board):
            row, col = divmod(i, 3)
            x, y = 64 + col * 44, 3 + row * 44
            pending = i == game.pending
            self.board.create_rectangle(x, y, x + 40, y + 40, width=0,
                                        fill="#def2e8" if pending else BG)
            if piece == "X":
                for a, b in ((11, 29), (29, 11)):
                    self.board.create_line(x + 11, y + a, x + 29, y + b, fill=INK, width=3)
            elif piece == "O" or pending:
                self.board.create_oval(x + 10, y + 10, x + 30, y + 30, outline=GREEN,
                                       width=3, dash=(3, 3) if pending else ())
