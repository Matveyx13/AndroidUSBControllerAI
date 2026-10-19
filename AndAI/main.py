"""USB Android controller with a scrcpy-only screen source and optional vision AI."""

from __future__ import annotations

import io
import json
import ctypes
import hashlib
import math
import os
import queue
import re
import shlex
import shutil
import threading
import time
import tkinter as tk
import xml.etree.ElementTree as ET
from ctypes import wintypes
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable

from PIL import Image, ImageGrab, ImageTk

from ai_agent import OpenAIVisionAgent
from dictionary_manager import download_dictionary, merge_vocabulary, read_dictionary_file
from local_agent import DOWNLOADED_DICTIONARY_PATH, LocalBrain, VOCABULARY, dictionary_matches, reload_dictionary
from ollama_agent import OllamaVisionAgent
from phone import PhoneController
from torch_agent import TorchVisionAgent


TASK_PLACEHOLDER = "Напишите сообщение ИИ или что сделать на телефоне…"


class PhoneApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("USB Android AI Controller")
        self.root.geometry("1160x850")
        self.root.minsize(980, 700)

        self.settings_path = Path(__file__).resolve().with_name("settings.json")
        self.settings = self._read_settings()
        self.phone = PhoneController()
        self.startup_error = ""
        saved_adb_path = str(self.settings.get("adb_path", ""))
        if saved_adb_path:
            try:
                self.phone.set_adb_path(saved_adb_path)
            except (OSError, ValueError) as error:
                self.startup_error = str(error)
        saved_scrcpy_path = str(self.settings.get("scrcpy_path", ""))
        self.scrcpy_startup_error = ""
        if saved_scrcpy_path:
            try:
                self.phone.set_scrcpy_path(saved_scrcpy_path)
            except (OSError, ValueError) as error:
                self.scrcpy_startup_error = str(error)

        self.events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.console_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.devices: dict[str, dict[str, str]] = {}
        self.preview_lock = threading.Lock()
        self.preview_image: Image.Image | None = None
        self.preview_sequence = 0
        self.displayed_preview_sequence = 0
        self.preview_stop_event = threading.Event()
        self.preview_stop_event.set()
        self.photo: ImageTk.PhotoImage | None = None
        self.frame_size = (0, 0)
        self.drag_start: tuple[int, int, float] | None = None
        self.capture_lock = threading.Lock()
        self.last_capture_error = ""
        self.last_preview_capture_error = ""
        self.scrcpy_overlay_hidden = False
        self.scrcpy_last_app_active: bool | None = None
        self.scrcpy_aspect_ratio: float | None = None
        self.scrcpy_sync_after_id: str | None = None
        self.debug_overlay_window: tk.Toplevel | None = None
        self.debug_overlay_canvas: tk.Canvas | None = None
        self.debug_overlay_hwnd: int | None = None
        self.debug_overlay_rect: tuple[int, int, int, int] | None = None
        self.debug_overlay_source_offset = (0, 0)
        self.debug_overlay_source_size = (0, 0)
        self.debug_overlay_photo: ImageTk.PhotoImage | None = None
        self.debug_overlay_visible = False
        self.debug_overlay_draw_key: tuple[Any, ...] | None = None
        self.debug_overlay_error = ""
        self.debug_overlay_color = "#010203"
        self.protected_zones = self._normalize_protected_zones(self.settings.get("protected_zones", []))
        self.zone_selection_active = False
        self.zone_selection_start: tuple[int, int] | None = None
        self.zone_selection_current: tuple[int, int] | None = None
        self.zone_button: ttk.Button | None = None
        self.zone_clear_button: ttk.Button | None = None
        self.zone_info: ttk.Label | None = None
        self.user32 = ctypes.WinDLL("user32", use_last_error=True) if os.name == "nt" else None
        if self.user32 is not None:
            self.user32.GetForegroundWindow.restype = wintypes.HWND
            self.user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            self.user32.GetAncestor.restype = wintypes.HWND
            self.user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
            self.user32.GetWindowRect.restype = wintypes.BOOL
            self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
            self.user32.GetClientRect.restype = wintypes.BOOL
            self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
            self.user32.ClientToScreen.restype = wintypes.BOOL
            self.user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            self.user32.SetWindowPos.argtypes = [
                wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, wintypes.UINT,
            ]
            self.user32.SetWindowPos.restype = wintypes.BOOL
            self.user32.SetLayeredWindowAttributes.argtypes = [
                wintypes.HWND, wintypes.DWORD, wintypes.BYTE, wintypes.DWORD,
            ]
            self.user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
        self.busy = False
        self.status_var = tk.StringVar(value="Подключите телефон кабелем USB и включите USB debugging")
        self.device_var = tk.StringVar()
        self.adb_path_var = tk.StringVar(value=self.phone.adb or "")
        self.scrcpy_path_var = tk.StringVar(value=self.phone.scrcpy or "")
        self.mode_var = tk.StringVar(value=self.settings.get("ai_mode", "local"))
        if self.mode_var.get() not in ("local", "api", "ollama", "torch"):
            self.mode_var.set("local")
        default_brain = str(Path(__file__).resolve().with_name("brain.json"))
        self.brain_path_var = tk.StringVar(value=self.settings.get("brain_path", default_brain))
        self.model_var = tk.StringVar(value=str(self.settings.get("model", "gpt-6-astra")))
        self.ollama_model_var = tk.StringVar(value=str(self.settings.get("ollama_model", "qwen3-vl:4b")))
        self.ollama_url_var = tk.StringVar(value=str(self.settings.get("ollama_url", "http://localhost:11434")))
        default_torch_brain = str(Path(__file__).resolve().with_name("torch_brain.pt"))
        self.torch_brain_path_var = tk.StringVar(value=str(self.settings.get("torch_brain_path", default_torch_brain)))
        default_screen_memory = str(Path(__file__).resolve().with_name("screen_memory.json"))
        self.screen_memory_path_var = tk.StringVar(
            value=str(self.settings.get("screen_memory_path", default_screen_memory))
        )
        self.screen_memory_path = self.screen_memory_path_var.get().strip()
        self.screen_memory_lock = threading.Lock()
        self.screen_memory: list[dict[str, Any]] = self._load_screen_memory()
        self.dictionary_url_var = tk.StringVar(value=str(self.settings.get("dictionary_url", "")))
        try:
            initial_action_delay = int(self.settings.get("action_delay_ms", 500))
        except (TypeError, ValueError, OverflowError):
            initial_action_delay = 500
        self.action_delay_var = tk.StringVar(value=str(max(100, min(5000, initial_action_delay))))
        self.action_delay_ms = max(100, min(5000, initial_action_delay))
        self.ai_mode_info_var = tk.StringVar()
        self.dictionary_preview_var = tk.StringVar()
        self.api_key_var = tk.StringVar()
        self.autonomous_var = tk.BooleanVar(value=False)
        self.curiosity_var = tk.BooleanVar(value=bool(self.settings.get("curiosity_mode", False)))
        self.autonomy_info_var = tk.StringVar()
        self.scrcpy_hwnd: int | None = None
        self.scrcpy_failed_serial: str | None = None
        self.learning_active = False
        self.learning_phrase = ""
        self.learning_actions: list[dict[str, Any]] = []
        self.learning_button: ttk.Button | None = None
        self.debug_button: ttk.Button | None = None
        self.debug_live = False
        self.debug_annotations: list[dict[str, Any]] = []
        self.debug_ui_annotations: list[dict[str, Any]] = []
        self.debug_ocr_annotations: list[dict[str, Any]] = []
        self.debug_ocr_thread: threading.Thread | None = None
        self.debug_ui_thread: threading.Thread | None = None
        self.debug_wakeup = threading.Event()
        self.debug_ui_wakeup = threading.Event()
        self.trace: tk.Text | None = None
        self.thought_view: tk.Text | None = None
        self.chat_messages: list[dict[str, str]] = []
        self.chat_view: tk.Text | None = None
        self.ai_stop_event = threading.Event()
        self.ai_busy = False
        self.ai_curiosity_active = False
        self.curiosity_button: ttk.Button | None = None
        self.material_icons: dict[tuple[str, str], ImageTk.PhotoImage] = {}
        self.torch_agent: TorchVisionAgent | None = None
        self.phone.action_listener = self._record_demo_action
        self.phone.ai_action_guard = self._guard_ai_action
        self.key_var = tk.StringVar(value="volume_up")
        self.last_screen_message = ""

        self._build_ui()
        self.root.bind("<Configure>", self._schedule_scrcpy_overlay_sync, add="+")
        self.screen_frame.bind("<Configure>", self._schedule_scrcpy_overlay_sync, add="+")
        self._update_ai_mode(save=False)
        self._refresh_dictionary_preview()
        if self.startup_error:
            self._append_log(f"Сохранённый путь ADB не применён: {self.startup_error}")
        if self.scrcpy_startup_error:
            self._append_log(f"Сохранённый путь scrcpy не применён: {self.scrcpy_startup_error}")
        if self.settings.get("brain_path") and not Path(self.brain_path_var.get()).is_file():
            self._append_log("Файл локального мозга пока не создан. Нажмите «Сохранить мозг» или обучите первый пример.")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._drain_events)
        threading.Thread(target=self._screen_loop, daemon=True).start()
        self.refresh_devices()
        self._queue_log(f"Готово: загружено RU/EN терминов — {len(VOCABULARY)}. Здесь отображаются наблюдения и шаги ИИ.")
        if self.screen_memory:
            self._queue_log(f"Память экранов: восстановлено записей — {len(self.screen_memory)}.")

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg="#f3f6fb")
        style.configure(".", font=("Segoe UI", 9))
        style.configure("TFrame", background="#f3f6fb")
        style.configure("TLabel", background="#f3f6fb", foreground="#202124")
        style.configure("TButton", padding=(9, 6), background="#ffffff", foreground="#202124")
        style.map("TButton", background=[("active", "#e8f0fe"), ("disabled", "#e8eaed")])
        style.configure("TEntry", padding=5)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#1a73e8")
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground="#3c4043")
        style.configure("Primary.TButton", padding=(12, 9), background="#1a73e8", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("Primary.TButton", background=[("active", "#1765cc"), ("disabled", "#9bbcf0")], foreground=[("disabled", "#ffffff")])
        style.configure("Research.TButton", padding=(10, 7), background="#e8f0fe", foreground="#174ea6", font=("Segoe UI", 9, "bold"))
        style.map("Research.TButton", background=[("active", "#d2e3fc")])
        style.configure("Danger.TButton", padding=(10, 7), background="#d93025", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("Danger.TButton", background=[("active", "#b3261e"), ("disabled", "#e6a09b")], foreground=[("disabled", "#ffffff")])
        self._load_material_icons()

        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Android через USB", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.status_var).pack(side="left", padx=18)
        ttk.Button(
            header, text="Обновить устройства", command=self.refresh_devices,
            image=self.material_icons.get(("blue", "refresh")), compound="left",
        ).pack(side="right")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left_panel = ttk.Frame(body, width=375)
        right = ttk.Frame(body, padding=8)
        body.add(left_panel, weight=0)
        body.add(right, weight=1)

        left_canvas = tk.Canvas(left_panel, highlightthickness=0, borderwidth=0)
        left_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scroll.pack(side="right", fill="y")
        left = ttk.Frame(left_canvas, padding=8)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda _event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda event: left_canvas.itemconfigure(left_window, width=event.width))
        left_canvas.bind("<MouseWheel>", lambda event: left_canvas.yview_scroll(int(-event.delta / 120), "units"))

        ttk.Label(left, text="USB устройство", style="Section.TLabel").pack(anchor="w")
        self.device_combo = ttk.Combobox(left, textvariable=self.device_var, state="readonly")
        self.device_combo.pack(fill="x", pady=(4, 5))
        self.device_combo.bind("<<ComboboxSelected>>", self._device_selected)
        ttk.Button(
            left, text="Подключить / запустить ADB", command=self.start_adb,
            image=self.material_icons.get(("blue", "devices")), compound="left",
        ).pack(fill="x", pady=(0, 10))

        ttk.Label(left, text="Путь к adb.exe", style="Section.TLabel").pack(anchor="w")
        adb_row = ttk.Frame(left)
        adb_row.pack(fill="x", pady=(3, 4))
        ttk.Entry(adb_row, textvariable=self.adb_path_var).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(adb_row, text="Обзор…", command=self.browse_adb).pack(side="right")
        ttk.Button(left, text="Применить путь к ADB", command=self.apply_adb_path).pack(fill="x", pady=(0, 8))

        ttk.Label(left, text="Путь к scrcpy.exe (источник экрана)", style="Section.TLabel").pack(anchor="w")
        scrcpy_row = ttk.Frame(left)
        scrcpy_row.pack(fill="x", pady=(3, 4))
        ttk.Entry(scrcpy_row, textvariable=self.scrcpy_path_var).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(scrcpy_row, text="Обзор", command=self.browse_scrcpy).pack(side="right")
        ttk.Button(left, text="Применить путь к scrcpy", command=self.apply_scrcpy_path).pack(fill="x", pady=(0, 8))

        ttk.Label(left, text="Кнопки телефона", style="Section.TLabel").pack(anchor="w", pady=(4, 3))
        keyrow = ttk.Frame(left)
        keyrow.pack(fill="x")
        ttk.Button(keyrow, text="Питание", image=self.material_icons.get(("blue", "power_settings_new")), compound="left", command=lambda: self._action("Нажатие Power", lambda: self.phone.press_key("power"))).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(keyrow, text="Громкость +", image=self.material_icons.get(("blue", "volume_up")), compound="left", command=lambda: self._action("Volume up", lambda: self.phone.press_key("volume_up"))).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(keyrow, text="Громкость −", image=self.material_icons.get(("blue", "volume_down")), compound="left", command=lambda: self._action("Volume down", lambda: self.phone.press_key("volume_down"))).pack(side="left", expand=True, fill="x", padx=(3, 0))

        navrow = ttk.Frame(left)
        navrow.pack(fill="x", pady=4)
        for label, key, icon_name in (("Назад", "back", "arrow_back"), ("Домой", "home", "home"), ("Недавние", "recents", "history")):
            ttk.Button(
                navrow, text=label, image=self.material_icons.get(("blue", icon_name)), compound="left",
                command=lambda l=label, k=key: self._action(l, lambda: self.phone.press_key(k)),
            ).pack(side="left", expand=True, fill="x", padx=2)

        holdrow = ttk.Frame(left)
        holdrow.pack(fill="x", pady=(2, 8))
        ttk.Label(holdrow, text="Долгое нажатие:").pack(side="left")
        ttk.Combobox(holdrow, textvariable=self.key_var, state="readonly", width=15, values=("power", "volume_up", "volume_down", "home", "back")).pack(side="left", padx=4)
        ttk.Button(holdrow, text="Отправить", command=self.hold_key).pack(side="left")

        ttk.Label(left, text="Интернет", style="Section.TLabel").pack(anchor="w")
        netrow = ttk.Frame(left)
        netrow.pack(fill="x", pady=(3, 8))
        ttk.Button(netrow, text="Wi-Fi вкл.", image=self.material_icons.get(("blue", "wifi")), compound="left", command=lambda: self._action("Wi-Fi", lambda: self.phone.set_network("wifi", True))).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(netrow, text="Wi-Fi выкл.", image=self.material_icons.get(("blue", "wifi")), compound="left", command=lambda: self._action("Wi-Fi", lambda: self.phone.set_network("wifi", False))).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(netrow, text="Моб. данные", image=self.material_icons.get(("blue", "wifi")), compound="left", command=self.toggle_mobile_data).pack(side="left", expand=True, fill="x", padx=(3, 0))

        ttk.Label(left, text="Bluetooth", style="Section.TLabel").pack(anchor="w")
        btrow = ttk.Frame(left)
        btrow.pack(fill="x", pady=(3, 8))
        ttk.Button(btrow, text="Включить", image=self.material_icons.get(("blue", "bluetooth")), compound="left", command=lambda: self._action("Bluetooth", lambda: self.phone.set_bluetooth(True))).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(btrow, text="Выключить", image=self.material_icons.get(("blue", "bluetooth")), compound="left", command=lambda: self._action("Bluetooth", lambda: self.phone.set_bluetooth(False))).pack(side="left", expand=True, fill="x", padx=(3, 0))

        ttk.Label(left, text="Загрузка / Fastboot", style="Section.TLabel").pack(anchor="w")
        bootrow = ttk.Frame(left)
        bootrow.pack(fill="x", pady=(3, 8))
        ttk.Button(bootrow, text="В bootloader", command=lambda: self._action("Bootloader", lambda: self.phone.reboot("fastboot"))).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(bootrow, text="В Recovery", command=lambda: self._action("Recovery", lambda: self.phone.reboot("recovery"))).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(bootrow, text="Перезагрузить", command=lambda: self._action("Перезагрузка", lambda: self.phone.reboot("system"))).pack(side="left", expand=True, fill="x", padx=(3, 0))

        textrow = ttk.Frame(left)
        textrow.pack(fill="x", pady=(0, 8))
        self.manual_text_var = tk.StringVar()
        ttk.Entry(textrow, textvariable=self.manual_text_var).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(textrow, text="Ввести текст", command=self.type_manual_text).pack(side="right")

        ttk.Separator(left).pack(fill="x", pady=6)
        ttk.Label(left, text="ИИ: правила, OpenAI или локальная модель", style="Section.TLabel").pack(anchor="w")
        mode_row = ttk.Frame(left)
        mode_row.pack(fill="x", pady=(2, 4))
        mode_options = (("Правила", "local"), ("OpenAI API", "api"), ("Ollama / Qwen", "ollama"), ("Torch / случайная модель", "torch"))
        for index, (label, value) in enumerate(mode_options):
            ttk.Radiobutton(mode_row, text=label, variable=self.mode_var, value=value, command=self._update_ai_mode).grid(
                row=index // 2, column=index % 2, sticky="w", padx=(0, 8), pady=1
            )
        ttk.Label(left, textvariable=self.ai_mode_info_var, wraplength=325).pack(anchor="w", pady=(0, 4))
        self.autonomous_check = ttk.Checkbutton(
            left,
            text="Полная автономия: разрешить shell и Fastboot",
            variable=self.autonomous_var,
        )
        self.autonomous_check.pack(anchor="w", pady=(0, 2))
        ttk.Label(
            left,
            textvariable=self.autonomy_info_var,
            wraplength=325,
        ).pack(anchor="w", pady=(0, 5))

        ttk.Label(left, text="Файл памяти локального мозга (JSON)").pack(anchor="w")
        brain_row = ttk.Frame(left)
        brain_row.pack(fill="x", pady=(2, 3))
        self.brain_entry = ttk.Entry(brain_row, textvariable=self.brain_path_var)
        self.brain_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(brain_row, text="Файл…", command=self.browse_brain).pack(side="right")
        brain_buttons = ttk.Frame(left)
        brain_buttons.pack(fill="x", pady=(0, 4))
        ttk.Button(brain_buttons, text="Обучить пример", command=self.teach_rule).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(brain_buttons, text="Сохранить мозг", command=self.save_brain).pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.learning_button = ttk.Button(left, text="▶ Самообучение: записать пример", command=self.toggle_learning)
        self.learning_button.pack(fill="x", pady=(0, 5))

        ttk.Label(left, text="OpenAI API key (не сохраняется на диск)").pack(anchor="w")
        self.api_key_entry = ttk.Entry(left, textvariable=self.api_key_var, show="•")
        self.api_key_entry.pack(fill="x", pady=(2, 4))
        ttk.Label(left, text="Модель OpenAI API").pack(anchor="w")
        self.model_entry = ttk.Entry(left, textvariable=self.model_var)
        self.model_entry.pack(fill="x", pady=(2, 4))
        ttk.Label(left, text="Адрес Ollama (локально)").pack(anchor="w")
        self.ollama_url_entry = ttk.Entry(left, textvariable=self.ollama_url_var)
        self.ollama_url_entry.pack(fill="x", pady=(2, 4))
        ttk.Label(left, text="Модель Ollama, например qwen3-vl:4b").pack(anchor="w")
        self.ollama_model_entry = ttk.Entry(left, textvariable=self.ollama_model_var)
        self.ollama_model_entry.pack(fill="x", pady=(2, 4))
        ttk.Label(left, text="Файл весов Torch-модели (.pt)").pack(anchor="w")
        torch_brain_row = ttk.Frame(left)
        torch_brain_row.pack(fill="x", pady=(2, 4))
        self.torch_brain_entry = ttk.Entry(torch_brain_row, textvariable=self.torch_brain_path_var)
        self.torch_brain_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(torch_brain_row, text="Открыть…", command=self.browse_torch_brain).pack(side="right")
        ttk.Button(left, text="Сохранить модель в новый .pt…", command=self.save_torch_brain_as).pack(fill="x", pady=(0, 5))
        ttk.Label(left, text="Память экранов ИИ (JSON)").pack(anchor="w")
        screen_memory_row = ttk.Frame(left)
        screen_memory_row.pack(fill="x", pady=(2, 5))
        ttk.Entry(screen_memory_row, textvariable=self.screen_memory_path_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(screen_memory_row, text="Файл…", command=self.browse_screen_memory).pack(side="right")
        ttk.Label(left, text="Сообщение ИИ или задача для телефона", style="Section.TLabel").pack(anchor="w")
        self.task_box = tk.Text(left, height=4, wrap="word", font=("Segoe UI", 9))
        self.task_box.pack(fill="x", pady=(2, 5))
        self.task_box.insert("1.0", TASK_PLACEHOLDER)
        self.task_box.bind("<KeyRelease>", self._refresh_dictionary_preview)
        self.task_box.bind("<FocusIn>", self._clear_task_placeholder)
        self.task_box.bind("<FocusOut>", self._restore_task_placeholder)
        ttk.Label(left, text="Словарь RU ↔ EN", style="Section.TLabel").pack(anchor="w")
        ttk.Label(left, textvariable=self.dictionary_preview_var, wraplength=325).pack(anchor="w", pady=(2, 5))
        ttk.Label(left, text="Скачать или импортировать словарь (JSON, TEI/XML, CSV, TXT)").pack(anchor="w")
        ttk.Entry(left, textvariable=self.dictionary_url_var).pack(fill="x", pady=(2, 3))
        dictionary_row = ttk.Frame(left)
        dictionary_row.pack(fill="x", pady=(0, 6))
        ttk.Button(dictionary_row, text="Скачать по URL", command=self.download_dictionary).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(dictionary_row, text="Импортировать файл…", command=self.import_dictionary_file).pack(side="left", expand=True, fill="x", padx=(3, 0))

        self.ai_button = ttk.Button(
            left, text="Отправить ИИ / выполнить", style="Primary.TButton",
            image=self.material_icons.get(("white", "play_arrow")), compound="left", command=self.run_ai,
        )
        self.ai_button.pack(fill="x", pady=(0, 6))
        curiosity_row = ttk.Frame(left)
        curiosity_row.pack(fill="x", pady=(0, 5))
        ttk.Checkbutton(
            curiosity_row,
            text="Режим любопытства",
            variable=self.curiosity_var,
            command=self._save_settings,
        ).pack(side="left", expand=True, anchor="w")
        self.curiosity_button = ttk.Button(
            curiosity_row, text="Исследовать экран", style="Research.TButton",
            image=self.material_icons.get(("blue", "play_arrow")), compound="left",
            command=self.run_curiosity,
        )
        self.curiosity_button.pack(side="right")
        delay_row = ttk.Frame(left)
        delay_row.pack(fill="x", pady=(0, 4))
        ttk.Label(delay_row, text="Пауза между действиями ИИ (мс)").pack(side="left")
        self.action_delay_spin = ttk.Spinbox(
            delay_row,
            from_=100,
            to=5000,
            increment=100,
            width=7,
            justify="right",
            textvariable=self.action_delay_var,
            command=self._commit_action_delay,
        )
        self.action_delay_spin.pack(side="right")
        self.action_delay_spin.bind("<FocusOut>", self._commit_action_delay)
        self.action_delay_spin.bind("<Return>", self._commit_action_delay)
        ttk.Label(
            left,
            text="Исследование работает до остановки. Разрешены любые действия, кроме вызовов экстренным службам.",
            wraplength=325,
        ).pack(anchor="w", pady=(0, 5))

        ttk.Label(left, text="Мысли и наблюдения ИИ", style="Section.TLabel").pack(anchor="w")
        self.thought_view = tk.Text(left, height=6, wrap="word", state="disabled", font=("Segoe UI", 9), bg="#eef4ff", relief="flat")
        self.thought_view.pack(fill="x", pady=(3, 6))

        ttk.Label(left, text="Наблюдения и выполненные шаги", style="Section.TLabel").pack(anchor="w")
        self.trace = tk.Text(left, height=6, wrap="word", state="disabled", font=("Consolas", 8))
        self.trace.pack(fill="x", pady=(3, 6))

        ttk.Label(left, text="Диалог с ИИ", style="Section.TLabel").pack(anchor="w")
        self.chat_view = tk.Text(left, height=6, wrap="word", state="disabled", font=("Segoe UI", 9))
        self.chat_view.pack(fill="x", pady=(3, 6))

        ttk.Label(left, text="Журнал", style="Section.TLabel").pack(anchor="w")
        self.log = tk.Text(left, height=8, wrap="word", state="disabled", font=("Consolas", 8))
        self.log.pack(fill="both", expand=True, pady=(3, 0))

        screen_header = ttk.Frame(right)
        screen_header.pack(fill="x")
        ttk.Label(screen_header, text="Экран телефона", style="Section.TLabel").pack(side="left")
        self.debug_button = ttk.Button(
            screen_header, text="Включить live-отладку", command=self.debug_screen,
            image=self.material_icons.get(("blue", "visibility")), compound="left",
        )
        self.debug_button.pack(side="right")
        ttk.Label(screen_header, text="Нажмите или перетащите мышь по кадру для управления").pack(side="right")
        zone_bar = ttk.Frame(right)
        zone_bar.pack(fill="x", pady=(3, 0))
        self.zone_button = ttk.Button(
            zone_bar, text="Выделить запретную зону", command=self.toggle_zone_selection,
        )
        self.zone_button.pack(side="left")
        self.zone_clear_button = ttk.Button(
            zone_bar, text="Очистить зоны", command=self.clear_protected_zones,
        )
        self.zone_clear_button.pack(side="left", padx=(5, 0))
        self.zone_info = ttk.Label(zone_bar, text=self._protected_zone_status())
        self.zone_info.pack(side="left", padx=(8, 0))
        self.screen_frame = ttk.Frame(right, relief="sunken")
        self.screen_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.screen_label = tk.Label(self.screen_frame, text="Ожидание scrcpy…", bg="#17191d", fg="#e7e7e7", font=("Segoe UI", 13))
        self.screen_label.pack(expand=True, fill="both")
        self.screen_info = ttk.Label(right, text="", anchor="center", wraplength=650)
        self.screen_info.pack(fill="x", pady=(4, 0))
        self.screen_label.bind("<ButtonPress-1>", self._screen_press)
        self.screen_label.bind("<ButtonRelease-1>", self._screen_release)

    def _load_material_icons(self) -> None:
        icon_dir = Path(__file__).resolve().parent / "assets" / "material-icons"
        for name in (
            "arrow_back", "bluetooth", "devices", "history", "home", "play_arrow", "power_settings_new",
            "refresh", "stop", "visibility", "volume_down", "volume_up", "wifi",
        ):
            path = icon_dir / f"{name}.png"
            if not path.is_file():
                continue
            try:
                source = Image.open(path).convert("RGBA")
                alpha = source.getchannel("A")
                for tint, color in (("blue", "#1a73e8"), ("white", "#ffffff")):
                    icon = Image.new("RGBA", source.size, color)
                    icon.putalpha(alpha)
                    self.material_icons[(tint, name)] = ImageTk.PhotoImage(icon)
            except OSError:
                continue

    def _read_settings(self) -> dict[str, Any]:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _normalize_protected_zones(value: Any) -> list[tuple[float, float, float, float]]:
        zones: list[tuple[float, float, float, float]] = []
        if not isinstance(value, list):
            return zones
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(part) for part in item)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(part) and 0.0 <= part <= 1.0 for part in (x1, y1, x2, y2)):
                continue
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            if right - left >= 0.003 and bottom - top >= 0.003:
                zones.append((left, top, right, bottom))
        return zones

    def _protected_zone_status(self) -> str:
        if self.zone_selection_active:
            return "Перетащи рамку; красные зоны блокируют действия ИИ"
        if self.protected_zones:
            return f"Запретных зон: {len(self.protected_zones)} · ручные касания работают"
        return "Перетащи рамку на экране телефона"

    def _update_zone_controls(self) -> None:
        if self.zone_button is not None:
            self.zone_button.configure(
                text="Завершить выделение" if self.zone_selection_active else "Выделить запретную зону"
            )
        if self.zone_clear_button is not None:
            self.zone_clear_button.configure(state="normal" if self.protected_zones else "disabled")
        if self.zone_info is not None:
            self.zone_info.configure(text=self._protected_zone_status())

    def toggle_zone_selection(self) -> None:
        if self.zone_selection_active:
            self._finish_zone_selection()
            return
        if self.learning_active:
            messagebox.showinfo("Защитные зоны", "Сначала останови запись самообучения.")
            return
        if (
            os.name != "nt" or self.user32 is None
            or self.scrcpy_hwnd is None or not self.phone.scrcpy_running
            or not self.phone.scrcpy_video_ready
        ):
            messagebox.showinfo("Защитные зоны", "Для выделения подключи телефон и запусти видеопоток scrcpy.")
            return
        self.zone_selection_active = True
        self.zone_selection_start = None
        self.zone_selection_current = None
        self.debug_overlay_draw_key = None
        if not self._ensure_debug_overlay():
            self.zone_selection_active = False
            self._update_zone_controls()
            return
        self._sync_debug_overlay()
        if not self.debug_overlay_visible:
            self.zone_selection_active = False
            self._update_zone_controls()
            self._queue_log("Не удалось начать выделение: слой поверх scrcpy сейчас недоступен.")
            return
        screenshot = self._capture_for_ai()
        image: Image.Image | None = None
        if screenshot:
            try:
                image = Image.open(io.BytesIO(screenshot)).convert("RGB")
            except OSError:
                image = None
        self._set_zone_selection_capture(True, image)
        self._update_zone_controls()
        self._queue_log("Выделение запретных зон: перетащи мышью прямоугольник по экрану; можно добавить несколько зон.")

    def _finish_zone_selection(self) -> None:
        self.zone_selection_active = False
        self.zone_selection_start = None
        self.zone_selection_current = None
        self._set_zone_selection_capture(False)
        self._update_zone_controls()
        self.debug_overlay_draw_key = None
        self._sync_debug_overlay()
        self._queue_log(f"Выделение завершено. Для ИИ закрыто зон: {len(self.protected_zones)}.")

    def _set_zone_selection_capture(self, enabled: bool, image: Image.Image | None = None) -> None:
        canvas = self.debug_overlay_canvas
        overlay = self.debug_overlay_window
        hwnd = self.debug_overlay_hwnd
        if canvas is None or overlay is None or hwnd is None or self.user32 is None:
            return
        key_color = self.debug_overlay_color
        try:
            if enabled:
                capture_bg = "#17191d"
                overlay.configure(bg=capture_bg)
                canvas.configure(bg=capture_bg)
                canvas.delete("selection_bg")
                self.debug_overlay_photo = None
                if image is not None and self.debug_overlay_rect is not None:
                    source_width, source_height = self.debug_overlay_source_size
                    if source_width > 0 and source_height > 0:
                        if image.size != (source_width, source_height):
                            image = image.resize((source_width, source_height), Image.Resampling.BILINEAR)
                        offset_x, offset_y = self.debug_overlay_source_offset
                        width, height = self.debug_overlay_rect[2:]
                        image = image.crop((offset_x, offset_y, offset_x + width, offset_y + height))
                        self.debug_overlay_photo = ImageTk.PhotoImage(image)
                        canvas.create_image(
                            0, 0, image=self.debug_overlay_photo, anchor="nw", tags="selection_bg",
                        )
                        canvas.tag_lower("selection_bg")
                style_index = -20
                get_style = getattr(self.user32, "GetWindowLongPtrW", None) or self.user32.GetWindowLongW
                set_style = getattr(self.user32, "SetWindowLongPtrW", None) or self.user32.SetWindowLongW
                get_style.argtypes = [wintypes.HWND, ctypes.c_int]
                get_style.restype = ctypes.c_ssize_t
                set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
                set_style.restype = ctypes.c_ssize_t
                style = int(get_style(wintypes.HWND(hwnd), style_index)) & ~0x00000020
                set_style(wintypes.HWND(hwnd), style_index, style)
            else:
                overlay.configure(bg=key_color)
                canvas.configure(bg=key_color)
                canvas.delete("selection_bg")
                self.debug_overlay_photo = None
                style_index = -20
                get_style = getattr(self.user32, "GetWindowLongPtrW", None) or self.user32.GetWindowLongW
                set_style = getattr(self.user32, "SetWindowLongPtrW", None) or self.user32.SetWindowLongW
                get_style.argtypes = [wintypes.HWND, ctypes.c_int]
                get_style.restype = ctypes.c_ssize_t
                set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
                set_style.restype = ctypes.c_ssize_t
                style = int(get_style(wintypes.HWND(hwnd), style_index)) | 0x00000020
                set_style(wintypes.HWND(hwnd), style_index, style)
            self.user32.SetLayeredWindowAttributes(wintypes.HWND(hwnd), 0x00030201, 0, 0x00000001)
            self.debug_overlay_draw_key = None
            if self.debug_overlay_rect is not None:
                self._draw_debug_overlay(self.debug_overlay_rect[2], self.debug_overlay_rect[3])
        except (OSError, tk.TclError, AttributeError, TypeError, ValueError) as error:
            self._queue_log(f"Не удалось переключить слой выделения: {error}")

    def _zone_selection_press(self, event: tk.Event) -> str:
        if not self.zone_selection_active:
            return "break"
        self.zone_selection_start = (int(event.x), int(event.y))
        self.zone_selection_current = self.zone_selection_start
        self.debug_overlay_draw_key = None
        if self.debug_overlay_rect is not None:
            self._draw_debug_overlay(self.debug_overlay_rect[2], self.debug_overlay_rect[3])
        return "break"

    def _zone_selection_motion(self, event: tk.Event) -> str:
        if self.zone_selection_active and self.zone_selection_start is not None:
            self.zone_selection_current = (int(event.x), int(event.y))
            self.debug_overlay_draw_key = None
            if self.debug_overlay_rect is not None:
                self._draw_debug_overlay(self.debug_overlay_rect[2], self.debug_overlay_rect[3])
        return "break"

    def _zone_selection_release(self, event: tk.Event) -> str:
        start = self.zone_selection_start
        source_width, source_height = self.debug_overlay_source_size
        if start is None or source_width <= 0 or source_height <= 0:
            return "break"
        end = (int(event.x), int(event.y))
        offset_x, offset_y = self.debug_overlay_source_offset
        left = max(0.0, min(1.0, (min(start[0], end[0]) + offset_x) / source_width))
        top = max(0.0, min(1.0, (min(start[1], end[1]) + offset_y) / source_height))
        right = max(0.0, min(1.0, (max(start[0], end[0]) + offset_x) / source_width))
        bottom = max(0.0, min(1.0, (max(start[1], end[1]) + offset_y) / source_height))
        self.zone_selection_start = None
        self.zone_selection_current = None
        if right - left >= 0.003 and bottom - top >= 0.003:
            self.protected_zones.append((left, top, right, bottom))
            self._save_settings()
            self._queue_log(
                f"Добавлена запретная для ИИ зона {len(self.protected_zones)}: "
                f"x={left:.3f}..{right:.3f}, y={top:.3f}..{bottom:.3f}."
            )
        else:
            self._queue_log("Зона не добавлена: выдели прямоугольник побольше.")
        self._update_zone_controls()
        self.debug_overlay_draw_key = None
        if self.debug_overlay_rect is not None:
            self._draw_debug_overlay(self.debug_overlay_rect[2], self.debug_overlay_rect[3])
        return "break"

    def clear_protected_zones(self) -> None:
        self.protected_zones.clear()
        self._save_settings()
        self._update_zone_controls()
        self.debug_overlay_draw_key = None
        self._sync_debug_overlay()
        self._queue_log("Все запретные зоны для ИИ удалены.")

    def _save_settings(self) -> None:
        self.settings.update({
            "adb_path": self.adb_path_var.get().strip(),
            "scrcpy_path": self.scrcpy_path_var.get().strip(),
            "ai_mode": self.mode_var.get(),
            "brain_path": self.brain_path_var.get().strip(),
            "model": self.model_var.get().strip(),
            "ollama_model": self.ollama_model_var.get().strip(),
            "ollama_url": self.ollama_url_var.get().strip(),
            "torch_brain_path": self.torch_brain_path_var.get().strip(),
            "screen_memory_path": self.screen_memory_path_var.get().strip(),
            "dictionary_url": self.dictionary_url_var.get().strip(),
            "curiosity_mode": self.curiosity_var.get(),
            "action_delay_ms": self._get_action_delay_ms(),
            "protected_zones": [list(zone) for zone in self.protected_zones],
        })
        try:
            self.settings_path.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            self._append_log(f"Не удалось сохранить настройки: {error}")

    def _get_action_delay_ms(self) -> int:
        try:
            value = int(float(self.action_delay_var.get().strip()))
        except (TypeError, ValueError, OverflowError):
            value = 500
        value = max(100, min(5000, value))
        self.action_delay_var.set(str(value))
        self.action_delay_ms = value
        return value

    def _commit_action_delay(self, _event: object = None) -> None:
        self._get_action_delay_ms()
        self._save_settings()

    def _update_ai_mode(self, save: bool = True) -> None:
        local_mode = self.mode_var.get() == "local"
        api_mode = self.mode_var.get() == "api"
        ollama_mode = self.mode_var.get() == "ollama"
        torch_mode = self.mode_var.get() == "torch"
        self.api_key_entry.configure(state="normal" if api_mode else "disabled")
        self.model_entry.configure(state="normal" if api_mode else "disabled")
        self.brain_entry.configure(state="normal" if local_mode else "disabled")
        self.ollama_url_entry.configure(state="normal" if ollama_mode else "disabled")
        self.ollama_model_entry.configure(state="normal" if ollama_mode else "disabled")
        self.torch_brain_entry.configure(state="normal" if torch_mode else "disabled")
        self.autonomous_check.configure(state="normal")
        if local_mode:
            self.ai_mode_info_var.set("Офлайн-правила из JSON, без ключа и без анализа изображения. Для исследования выберите Torch или vision-модель.")
            self.autonomy_info_var.set("Локальные правила выполняют команды без API; для понимания экрана нужен OpenAI, Ollama или обученная vision-модель.")
        elif api_mode:
            self.ai_mode_info_var.set("OpenAI vision анализирует кадры; API-ключ используется только в памяти приложения.")
            self.autonomy_info_var.set("Доступны ADB-действия; полный shell/Fastboot включается отдельным флажком.")
        elif ollama_mode:
            self.ai_mode_info_var.set("Локальная vision-модель через Ollama. Нужны запущенный Ollama и vision-модель с поддержкой tools.")
            self.autonomy_info_var.set("Например, Qwen3-VL видит экран и вызывает ADB-инструменты локально, без API-ключа.")
        else:
            self.ai_mode_info_var.set("Создаётся локальная CNN со случайными весами, без API и скачивания предобученной модели. Для исследования включите любопытство.")
            self.autonomy_info_var.set("Пока модель не понимает текст и смысл кнопок. Она пробует доступные действия и учится на изменениях экрана; веса сохраняются в отдельный .pt-файл.")
        if save:
            self._save_settings()

    def _refresh_dictionary_preview(self, _event: object = None) -> None:
        if not hasattr(self, "task_box"):
            return
        task = self.task_box.get("1.0", "end").strip()
        matches = dictionary_matches(task)
        if matches:
            preview = ", ".join(matches[:6])
            if len(matches) > 6:
                preview += f" … (+{len(matches) - 6})"
            self.dictionary_preview_var.set(preview)
        else:
            self.dictionary_preview_var.set(f"Загружено терминов: {len(VOCABULARY)} · совпадения появятся при вводе задачи")

    def _clear_task_placeholder(self, _event: object = None) -> None:
        if self.task_box.get("1.0", "end").strip() == TASK_PLACEHOLDER:
            self.task_box.delete("1.0", "end")
            self._refresh_dictionary_preview()

    def _restore_task_placeholder(self, _event: object = None) -> None:
        if not self.task_box.get("1.0", "end").strip():
            self.task_box.insert("1.0", TASK_PLACEHOLDER)
            self._refresh_dictionary_preview()

    def import_dictionary_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Импортировать словарь",
            filetypes=(("Словари", "*.json *.xml *.tei *.csv *.txt *.dict"), ("Все файлы", "*.*")),
        )
        if not selected:
            return
        self._start_dictionary_import(
            f"импорт {Path(selected).name}",
            lambda: read_dictionary_file(selected),
        )

    def download_dictionary(self) -> None:
        url = self.dictionary_url_var.get().strip()
        if not url:
            messagebox.showinfo("Скачать словарь", "Введите прямую ссылку на JSON, TEI/XML, CSV или TXT файл.")
            return
        self._save_settings()
        self._start_dictionary_import(f"скачивание {url}", lambda: download_dictionary(url))

    def _start_dictionary_import(self, label: str, loader: Callable[[], dict[str, list[str]]]) -> None:
        self._queue_log(f"Словарь: начато {label}")

        def work() -> None:
            try:
                vocabulary = loader()
                total = merge_vocabulary(DOWNLOADED_DICTIONARY_PATH, vocabulary)
                reload_dictionary()
                self.events.put(("dictionary_loaded", len(vocabulary), total, label))
            except Exception as error:
                self.events.put(("dictionary_error", str(error), label))

        threading.Thread(target=work, daemon=True).start()

    def _append_chat(self, role: str, content: str) -> None:
        if role in ("Вы", "ИИ"):
            self.chat_messages.append({"role": "user" if role == "Вы" else "assistant", "content": content})
            self.chat_messages = self.chat_messages[-40:]
        if self.chat_view is None:
            return
        self.chat_view.configure(state="normal")
        self.chat_view.insert("end", f"{role}: {content}\n\n")
        line_count = int(self.chat_view.index("end-1c").split(".", 1)[0])
        if line_count > 160:
            self.chat_view.delete("1.0", f"{line_count - 120}.0")
        self.chat_view.see("end")
        self.chat_view.configure(state="disabled")

    def run_curiosity(self) -> None:
        if self.ai_busy:
            if self.ai_curiosity_active:
                self.stop_ai()
            return
        self.curiosity_var.set(True)
        if self.mode_var.get() == "local":
            self.mode_var.set("torch")
            self._update_ai_mode(save=False)
        self._save_settings()
        self.run_ai(
            task_override=(
                "Самостоятельно исследуй Android-интерфейс и выполняй действия по увиденному экрану. "
                "Разрешено всё, кроме инициирования вызовов экстренным службам."
            ),
            force_curiosity=True,
        )

    def browse_adb(self) -> None:
        selected = filedialog.askopenfilename(
            title="Выберите adb.exe",
            filetypes=(("ADB executable", "*.exe"), ("Все файлы", "*.*")),
        )
        if selected:
            self.adb_path_var.set(selected)
            self.apply_adb_path()

    def apply_adb_path(self) -> None:
        try:
            selected = self.phone.set_adb_path(self.adb_path_var.get())
            self.adb_path_var.set(selected)
            self._save_settings()
            self._append_log(f"ADB executable: {selected or 'не найден'}")
            if self.phone.fastboot:
                self._append_log(f"Fastboot executable: {self.phone.fastboot}")
            self.refresh_devices()
        except (OSError, ValueError) as error:
            messagebox.showerror("Путь к ADB", str(error))

    def browse_scrcpy(self) -> None:
        selected = filedialog.askopenfilename(
            title="Выберите scrcpy.exe",
            filetypes=(("scrcpy executable", "scrcpy.exe"), ("Все файлы", "*.*")),
        )
        if selected:
            self.scrcpy_path_var.set(selected)
            self.apply_scrcpy_path()

    def apply_scrcpy_path(self) -> None:
        try:
            selected = self.phone.set_scrcpy_path(self.scrcpy_path_var.get())
            self.scrcpy_path_var.set(selected)
            self._save_settings()
            self.scrcpy_failed_serial = None
            if selected:
                self._append_log(f"scrcpy: {selected}")
            else:
                self._append_log("scrcpy не найден; отображение экрана и визуальный анализ недоступны")
            self.events.put(("scrcpy_stopped",))
        except (OSError, ValueError) as error:
            messagebox.showerror("Путь к scrcpy", str(error))

    def browse_torch_brain(self) -> None:
        current_path = Path(self.torch_brain_path_var.get() or "torch_brain.pt").expanduser()
        initial_dir = current_path.parent if current_path.parent.is_dir() else Path(__file__).resolve().parent
        selected = filedialog.askopenfilename(
            title="Загрузить веса Torch-модели",
            initialdir=str(initial_dir),
            filetypes=(("Веса PyTorch", "*.pt"), ("Все файлы", "*.*")),
        )
        if selected:
            self.torch_brain_path_var.set(selected)
            self._save_settings()
            self._append_log(f"Torch: для следующего запуска выбраны веса {Path(selected).name}.")

    def save_torch_brain_as(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Выберите путь для Torch-модели",
            defaultextension=".pt",
            initialfile=Path(self.torch_brain_path_var.get() or "torch_brain.pt").name,
            filetypes=(("Веса PyTorch", "*.pt"), ("Все файлы", "*.*")),
        )
        if selected:
            self.torch_brain_path_var.set(selected)
            self._save_settings()
            self._append_log(f"Torch: веса будут загружаться и сохраняться в {selected}.")

    def browse_screen_memory(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Файл памяти экранов",
            defaultextension=".json",
            initialfile=Path(self.screen_memory_path_var.get() or "screen_memory.json").name,
            filetypes=(("Память экранов JSON", "*.json"), ("Все файлы", "*.*")),
        )
        if selected:
            self.screen_memory_path_var.set(selected)
            with self.screen_memory_lock:
                self.screen_memory_path = selected
                self.screen_memory = self._read_screen_memory_file(Path(selected))
            self._save_settings()
            self._append_log(f"Память экранов: {len(self.screen_memory)} сохранённых экранов из {selected}.")

    def browse_brain(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Выберите файл локального мозга",
            defaultextension=".json",
            initialfile=Path(self.brain_path_var.get() or "brain.json").name,
            filetypes=(("Файл памяти JSON", "*.json"), ("Все файлы", "*.*")),
        )
        if selected:
            self.brain_path_var.set(selected)
            self._save_settings()

    def save_brain(self) -> None:
        path = self.brain_path_var.get().strip()
        if not path:
            messagebox.showerror("Файл мозга", "Укажите путь к файлу JSON.")
            return
        try:
            brain = LocalBrain(path)
            brain.save()
            self._save_settings()
            self._append_log(f"Мозг сохранён: {path} ({len(brain.rules)} правил)")
        except (OSError, ValueError) as error:
            messagebox.showerror("Не удалось сохранить мозг", str(error))

    def teach_rule(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Обучить локальный мозг")
        window.geometry("480x430")
        window.transient(self.root)
        window.grab_set()
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Фраза, которую мозг должен запомнить").pack(anchor="w")
        phrase_var = tk.StringVar()
        ttk.Entry(frame, textvariable=phrase_var).pack(fill="x", pady=(3, 10))
        ttk.Label(frame, text="Действия: по одной команде на строку").pack(anchor="w")
        script_box = tk.Text(frame, height=9, wrap="word", font=("Consolas", 10))
        script_box.pack(fill="both", expand=True, pady=(3, 5))
        ttk.Label(
            frame,
            text="Примеры: tap 500 900 · swipe 500 1400 500 400 500 · press power · wifi off · reboot fastboot · type Привет · shell getprop ro.product.model · fastboot getvar product",
            wraplength=450,
        ).pack(anchor="w", pady=(0, 8))

        def save_example() -> None:
            try:
                brain = LocalBrain(self.brain_path_var.get().strip())
                count = brain.add_rule(
                    phrase_var.get(),
                    script_box.get("1.0", "end"),
                    allow_autonomous=self.autonomous_var.get(),
                )
                self._save_settings()
                self._append_log(f"Локальный мозг: сохранено правил — {count}")
                window.destroy()
            except (OSError, ValueError) as error:
                messagebox.showerror("Пример не сохранён", str(error), parent=window)

        ttk.Button(frame, text="Запомнить пример", command=save_example).pack(fill="x")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {text}\n")
        line_count = int(self.log.index("end-1c").split(".", 1)[0])
        if line_count > 1200:
            self.log.delete("1.0", f"{line_count - 1000}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _queue_log(self, text: str) -> None:
        try:
            with self.console_lock:
                print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
        except Exception:
            # Tk's in-app log remains available when launched without a console.
            pass
        self.events.put(("log", text))
        self.events.put(("trace", text))

    def _queue_internal_log(self, text: str) -> None:
        """Send technical diagnostics to the UI without mixing them into console narration."""
        self.events.put(("log", text))
        self.events.put(("trace", text))

    def _read_screen_memory_file(self, path: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("screens", []) if isinstance(payload, dict) else []
            return [record for record in records if isinstance(record, dict)][-500:]
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            return []

    def _load_screen_memory(self) -> list[dict[str, Any]]:
        return self._read_screen_memory_file(Path(self.screen_memory_path).expanduser())

    def _remember_screen(
        self,
        frame: bytes,
        observation: str,
        terms: list[str],
        action: str | None,
    ) -> dict[str, Any]:
        """Persist a compact visual/text fingerprint and return a memory lookup result."""
        image = Image.open(io.BytesIO(frame)).convert("L").resize((24, 40), Image.Resampling.BILINEAR)
        signature = hashlib.sha256(image.tobytes()).hexdigest()[:20]
        normalized = " ".join(re.findall(r"\w+", observation.casefold(), flags=re.UNICODE))[:600]
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.screen_memory_lock:
            matched: dict[str, Any] | None = None
            for record in reversed(self.screen_memory):
                if record.get("signature") == signature:
                    matched = record
                    break
                prior_text = str(record.get("normalized_text", ""))
                if len(normalized) >= 20 and prior_text:
                    if prior_text == normalized:
                        matched = record
                        break
                    prior_words, current_words = set(prior_text.split()), set(normalized.split())
                    overlap = len(prior_words & current_words) / max(1, len(prior_words | current_words))
                    if overlap >= 0.88:
                        matched = record
                        break

            known = matched is not None
            prior_action = str(matched.get("last_action", "")) if matched else ""
            dirty = matched is None
            if matched is None:
                matched = {
                    "signature": signature,
                    "normalized_text": normalized,
                    "text": observation[:600],
                    "terms": terms[:12],
                    "first_seen": now,
                    "last_seen": now,
                    "visits": 0,
                    "last_action": "",
                }
                self.screen_memory.append(matched)
            dirty = dirty or matched.get("text") != observation[:600] or matched.get("terms") != terms[:12]
            matched.update({
                "signature": signature,
                "normalized_text": normalized,
                "text": observation[:600],
                "terms": terms[:12],
                "last_seen": now,
                "visits": int(matched.get("visits", 0)) + 1,
            })
            if action:
                dirty = dirty or matched.get("last_action") != action[:180]
                matched["last_action"] = action[:180]
            dirty = dirty or matched["visits"] % 10 == 0
            self.screen_memory = self.screen_memory[-500:]
            if not dirty:
                return {"known": known, "visits": matched["visits"], "last_action": prior_action}
            memory_path = Path(self.screen_memory_path).expanduser()
            try:
                memory_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = memory_path.with_suffix(memory_path.suffix + ".tmp")
                temp_path.write_text(
                    json.dumps({"version": 1, "screens": self.screen_memory}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temp_path.replace(memory_path)
            except OSError as error:
                return {"known": known, "visits": matched["visits"], "last_action": prior_action, "error": str(error)}
            return {"known": known, "visits": matched["visits"], "last_action": prior_action}

    @staticmethod
    def _screen_thought(
        observation: str,
        terms: list[str],
        previous_observation: str | None,
        action: str | None,
        memory: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        if observation:
            parts.append(f"На экране читаю: «{observation[:220]}».")
        else:
            parts.append("На этом кадре OCR не распознал надписи, поэтому описываю только изображение без догадок о его смысле.")
        if terms:
            parts.append("Словарь связывает найденные слова с понятиями: " + ", ".join(terms[:8]) + ".")
        else:
            parts.append("Совпадений с RU/EN словарём на распознанном тексте нет.")
        if action:
            if observation and previous_observation and observation == previous_observation:
                parts.append(f"После действия «{action}» распознанный текст не изменился.")
            elif observation:
                parts.append(f"После действия «{action}» экран показывает этот текст.")
            else:
                parts.append(f"После действия «{action}» проверяю следующий кадр; надписи пока не читаются.")
        if memory.get("known"):
            count = int(memory.get("visits", 1))
            parts.append(f"Я уже сохранял похожий экран; теперь он встречен {count} раз.")
            last_action = str(memory.get("last_action", ""))
            if last_action:
                parts.append(f"В прошлый раз здесь было действие: {last_action}.")
        elif not memory.get("error"):
            parts.append("Сохранил этот экран в долговременную память.")
        if memory.get("error"):
            parts.append(f"Не удалось записать память на диск: {memory['error']}.")
        return " ".join(parts)

    def _append_trace(self, text: str) -> None:
        if self.trace is None:
            return
        self.trace.configure(state="normal")
        self.trace.insert("end", f"{time.strftime('%H:%M:%S')}  {text}\n")
        line_count = int(self.trace.index("end-1c").split(".", 1)[0])
        if line_count > 120:
            self.trace.delete("1.0", f"{line_count - 100}.0")
        self.trace.see("end")
        self.trace.configure(state="disabled")

    def _queue_thought(self, text: str) -> None:
        try:
            with self.console_lock:
                print(f"ИИ: {text}", flush=True)
        except Exception:
            pass
        self.events.put(("thought", text))

    def _append_thought(self, text: str) -> None:
        if self.thought_view is None:
            return
        self.thought_view.configure(state="normal")
        self.thought_view.insert("end", f"{text}\n")
        line_count = int(self.thought_view.index("end-1c").split(".", 1)[0])
        if line_count > 240:
            self.thought_view.delete("1.0", f"{line_count - 200}.0")
        self.thought_view.see("end")
        self.thought_view.configure(state="disabled")

    @staticmethod
    def _describe_ai_action(name: str, arguments: dict[str, Any]) -> str:
        if name == "tap":
            return f"нажал на экран в точке ({arguments.get('x')}, {arguments.get('y')})"
        if name == "hold_screen":
            return (
                f"удерживал точку ({arguments.get('x')}, {arguments.get('y')}) "
                f"на {arguments.get('duration_ms', 1000)} мс"
            )
        if name == "swipe":
            return (
                f"провёл свайп от ({arguments.get('x1')}, {arguments.get('y1')}) "
                f"к ({arguments.get('x2')}, {arguments.get('y2')})"
            )
        if name == "press_key":
            labels = {
                "back": "Назад", "home": "Домой", "recents": "Недавние",
                "power": "Питание", "volume_up": "Громкость +", "volume_down": "Громкость −",
                "enter": "Ввод", "menu": "Меню", "wakeup": "Пробуждение", "camera": "Камера",
            }
            key = labels.get(str(arguments.get("key", "")), str(arguments.get("key", "кнопка")))
            suffix = " с удержанием" if arguments.get("long_press") else ""
            return f"нажал кнопку «{key}»{suffix}"
        if name == "type_text":
            return "ввёл текст в активное поле"
        if name == "open_app":
            return f"открыл приложение {arguments.get('package_name', '')}"
        if name == "set_network":
            feature = "Wi‑Fi" if arguments.get("feature") == "wifi" else "мобильных данных"
            state = "включил" if arguments.get("enabled") else "выключил"
            return f"{state} {feature}"
        if name == "set_bluetooth":
            return "включил Bluetooth" if arguments.get("enabled") else "выключил Bluetooth"
        if name == "reboot":
            targets = {"system": "Android", "recovery": "Recovery", "fastboot": "bootloader/Fastboot"}
            return f"перезагрузил устройство в {targets.get(str(arguments.get('target')), 'указанный режим')}"
        if name == "run_adb_shell":
            return "выполнил команду ADB shell"
        if name == "run_fastboot":
            return "выполнил команду Fastboot"
        return f"вызов инструмента {name}"

    def _record_demo_action(self, name: str, arguments: dict[str, Any]) -> None:
        if not self.learning_active:
            return
        if name not in {"tap", "swipe", "hold_screen", "press_key", "open_app", "set_network", "set_bluetooth"}:
            return
        if name == "press_key" and arguments.get("key") not in {"power", "volume_up", "volume_down", "home", "back", "recents"}:
            return
        self.learning_actions.append({"name": name, "arguments": dict(arguments)})
        self.events.put(("learning_count", len(self.learning_actions)))
        self._queue_log(f"Самообучение: записано действие {name} {arguments}")

    def toggle_learning(self) -> None:
        if self.learning_active:
            actions = self.learning_actions[:]
            phrase = self.learning_phrase
            self.learning_active = False
            self.learning_actions.clear()
            self.learning_phrase = ""
            if self.learning_button is not None:
                self.learning_button.configure(text="▶ Самообучение: записать пример")
            self.scrcpy_failed_serial = None
            if not actions:
                self._queue_log("Самообучение остановлено: действий не записано")
                return
            try:
                brain = LocalBrain(self.brain_path_var.get().strip())
                count = brain.add_actions(phrase, actions)
                self._save_settings()
                self._queue_log(f"Самообучение: пример «{phrase}» сохранён ({len(actions)} действий, {count} правил)")
            except (OSError, ValueError) as error:
                messagebox.showerror("Пример не сохранён", str(error))
            return

        phrase = simpledialog.askstring(
            "Самообучение",
            "Какой фразой запускать записанную последовательность?",
            parent=self.root,
        )
        if not phrase or len(phrase.strip()) < 2:
            if phrase:
                messagebox.showerror("Самообучение", "Введите фразу длиной хотя бы два символа.")
            return
        if self.debug_live:
            messagebox.showinfo("Самообучение", "Сначала выключите live-отладку экрана.")
            return
        if not self.brain_path_var.get().strip():
            messagebox.showerror("Файл памяти", "Укажите путь к JSON-файлу мозга.")
            return
        if not self.phone.serial or self.phone.state != "device":
            messagebox.showerror("Нет телефона", "Для записи подключите и выберите Android с активным ADB.")
            return
        try:
            LocalBrain(self.brain_path_var.get().strip())
        except (OSError, ValueError) as error:
            messagebox.showerror("Файл памяти", str(error))
            return
        self.learning_phrase = phrase.strip()
        self.learning_actions = []
        self.learning_active = True
        if self.learning_button is not None:
            self.learning_button.configure(text="■ Остановить и сохранить")
        self._queue_log(
            f"Самообучение началось для фразы «{self.learning_phrase}»; scrcpy остаётся источником экрана. "
            "Записываются действия, отправленные через AndAI; прямые касания в окне scrcpy не попадают в пример."
        )

    def refresh_devices(self) -> None:
        threading.Thread(target=self._discover_worker, daemon=True).start()

    def _discover_worker(self) -> None:
        try:
            devices, errors = self.phone.discover()
            self.events.put(("devices", devices, errors))
        except Exception as error:
            self.events.put(("log", f"ADB: {error}"))

    def _set_devices(self, devices: list[dict[str, str]], errors: list[str]) -> None:
        old = self.device_var.get().split("  [", 1)[0]
        self.devices = {f"{d['serial']}  [{d['kind']}: {d['state']}]": d for d in devices}
        self.device_combo["values"] = list(self.devices)
        choice = next((key for key in self.devices if key.split("  [", 1)[0] == old), None)
        if not choice and self.devices:
            choice = next(iter(self.devices))
        self.device_var.set(choice or "")
        if choice:
            device = self.devices[choice]
            self.phone.select(device["serial"], device["state"])
            if device["state"] not in ("device", "fastboot"):
                self.screen_info.configure(text=f"Статус ADB: {device['state']}")
            self.status_var.set(f"{device['kind']}: {device['state']} — {device['serial']}")
        elif errors:
            self.phone.select("", "offline")
            self.status_var.set(errors[0])
        else:
            self.phone.select("", "offline")
            self.status_var.set("Телефон не найден. Включите USB debugging и подтвердите разрешение на экране.")
        for error in errors:
            self._append_log(error)

    def _device_selected(self, _event: object = None) -> None:
        item = self.devices.get(self.device_var.get())
        if item:
            self.phone.select(item["serial"], item["state"])
            self.status_var.set(f"{item['kind']}: {item['state']} — {item['serial']}")

    def type_manual_text(self) -> None:
        text = self.manual_text_var.get()
        if text:
            self._action("Ввод текста", lambda: self.phone.type_text(text))

    def start_adb(self) -> None:
        def run() -> str:
            if not self.phone.adb:
                raise RuntimeError("Не найден adb. Укажите путь к adb.exe выше или установите Android SDK Platform-Tools.")
            result = self.phone._run([self.phone.adb, "start-server"], timeout=12)
            if result.returncode:
                raise RuntimeError(self.phone._decode(result.stderr) or "Не удалось запустить ADB server")
            return "ADB server запущен; подтвердите разрешение USB debugging на телефоне, если появится запрос"
        self._action("Подключение", run, then_refresh=True)

    def _action(self, title: str, operation: Callable[[], str], then_refresh: bool = False) -> None:
        def work() -> None:
            try:
                result = operation()
                self.events.put(("log", f"{title}: {result}"))
                self.events.put(("status", result))
            except Exception as error:
                self.events.put(("log", f"{title}: ошибка — {error}"))
                self.events.put(("status", str(error)))
            if then_refresh:
                devices, errors = self.phone.discover()
                self.events.put(("devices", devices, errors))
        threading.Thread(target=work, daemon=True).start()

    def hold_key(self) -> None:
        self._action("Долгое нажатие", lambda: self.phone.press_key(self.key_var.get(), long_press=True))
        self._queue_log(f"Для {self.key_var.get()} отправлен синтетический Android long-press event")

    def toggle_mobile_data(self) -> None:
        # There is no universal read permission for this setting; offer a direct toggle action.
        choice = messagebox.askyesnocancel("Мобильные данные", "Включить мобильные данные?\nДа — включить, Нет — выключить")
        if choice is not None:
            self._action("Мобильные данные", lambda: self.phone.set_network("mobile_data", bool(choice)))

    def _screen_press(self, event: tk.Event) -> None:
        if self.frame_size[0] > 0:
            self.drag_start = (event.x, event.y, time.monotonic())

    def _screen_release(self, event: tk.Event) -> None:
        if not self.drag_start or not self.frame_size[0] or not self.photo:
            return
        x0, y0, started = self.drag_start
        self.drag_start = None
        original_w, original_h = self.frame_size
        shown_w, shown_h = self.photo.width(), self.photo.height()
        if shown_w <= 0 or shown_h <= 0:
            return
        offset_x = max(0, (self.screen_label.winfo_width() - shown_w) // 2)
        offset_y = max(0, (self.screen_label.winfo_height() - shown_h) // 2)
        view_x0, view_y0 = x0 - offset_x, y0 - offset_y
        view_x1, view_y1 = event.x - offset_x, event.y - offset_y
        if not (0 <= view_x0 < shown_w and 0 <= view_y0 < shown_h and 0 <= view_x1 < shown_w and 0 <= view_y1 < shown_h):
            return
        x0 = max(0, min(original_w - 1, round(view_x0 * original_w / shown_w)))
        y0 = max(0, min(original_h - 1, round(view_y0 * original_h / shown_h)))
        x1 = max(0, min(original_w - 1, round(view_x1 * original_w / shown_w)))
        y1 = max(0, min(original_h - 1, round(view_y1 * original_h / shown_h)))
        elapsed = int((time.monotonic() - started) * 1000)
        if abs(view_x1 - view_x0) > 8 or abs(view_y1 - view_y0) > 8:
            self._action("Свайп", lambda: self.phone.swipe(x0, y0, x1, y1, elapsed))
        elif elapsed >= 500:
            self._action("Удержание экрана", lambda: self.phone.hold_screen(x0, y0, elapsed))
        else:
            self._action("Нажатие", lambda: self.phone.tap(x0, y0))

    @staticmethod
    def _find_scrcpy_window(process_id: int, expected_title: str) -> int | None:
        if os.name != "nt":
            return None
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        enum_callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        found: list[int] = []

        @enum_callback_type
        def inspect(window: int, _lparam: int) -> bool:
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
            if owner.value != process_id:
                return True
            length = user32.GetWindowTextLengthW(window)
            title_buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(window, title_buffer, len(title_buffer))
            if title_buffer.value == expected_title:
                found.append(int(window))
                return False
            return True

        user32.EnumWindows.argtypes = [enum_callback_type, wintypes.LPARAM]
        user32.EnumWindows.restype = wintypes.BOOL
        user32.EnumWindows(inspect, 0)
        return found[0] if found else None

    def _wait_for_scrcpy_window(self, process_id: int, title: str) -> int | None:
        for _ in range(100):
            if (
                self.stop_event.is_set()
                or not self.phone.scrcpy_running
                or self.phone.scrcpy_serial != self.phone.serial
                or self.phone.state != "device"
            ):
                return None
            window = self._find_scrcpy_window(process_id, title)
            if window is not None:
                return window
            if self.stop_event.wait(0.1):
                return None
        return None

    def _attach_scrcpy_window(self, window: int) -> None:
        self.scrcpy_hwnd = window
        self.last_preview_capture_error = ""
        self.scrcpy_overlay_hidden = False
        self.scrcpy_last_app_active = None
        self.scrcpy_aspect_ratio = None
        self.screen_info.configure(text="Нативный поток scrcpy • выравнивается по панели AndAI")
        self.status_var.set("scrcpy подключен")
        self.last_screen_message = ""
        self.root.after_idle(self._sync_scrcpy_overlay)
        self._start_scrcpy_preview(window)

    def _ensure_debug_overlay(self) -> bool:
        if os.name != "nt" or self.user32 is None:
            return False
        if self.debug_overlay_window is not None:
            try:
                if self.debug_overlay_window.winfo_exists():
                    return True
            except tk.TclError:
                pass
        overlay: tk.Toplevel | None = None
        try:
            overlay = tk.Toplevel(self.root)
            overlay.withdraw()
            overlay.overrideredirect(True)
            overlay.configure(bg=self.debug_overlay_color)
            overlay.wm_attributes("-transparentcolor", self.debug_overlay_color)
            overlay.wm_attributes("-topmost", True)
            canvas = tk.Canvas(
                overlay,
                bg=self.debug_overlay_color,
                highlightthickness=0,
                bd=0,
            )
            canvas.pack(fill="both", expand=True)
            canvas.bind("<ButtonPress-1>", self._zone_selection_press)
            canvas.bind("<B1-Motion>", self._zone_selection_motion)
            canvas.bind("<ButtonRelease-1>", self._zone_selection_release)
            overlay.update_idletasks()
            hwnd = int(overlay.winfo_id())
            get_style = getattr(self.user32, "GetWindowLongPtrW", None)
            set_style = getattr(self.user32, "SetWindowLongPtrW", None)
            if get_style is None:
                get_style = self.user32.GetWindowLongW
                set_style = self.user32.SetWindowLongW
            get_style.argtypes = [wintypes.HWND, ctypes.c_int]
            get_style.restype = ctypes.c_ssize_t
            set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
            set_style.restype = ctypes.c_ssize_t
            ex_style = int(get_style(wintypes.HWND(hwnd), -20))
            # Tk may rewrite extended styles while mapping a Toplevel. Apply them
            # again after deiconify so the mapped overlay stays click-through.
            ex_style |= 0x00080000 | 0x00000020 | 0x00000080 | 0x08000000
            set_style(wintypes.HWND(hwnd), -20, ex_style)
            overlay.deiconify()
            overlay.update_idletasks()
            ex_style = int(get_style(wintypes.HWND(hwnd), -20))
            ex_style |= 0x00080000 | 0x00000020 | 0x00000080 | 0x08000000
            set_style(wintypes.HWND(hwnd), -20, ex_style)
            # LWA_COLORKEY keeps only drawn annotations visible. Combined with
            # WS_EX_TRANSPARENT this lets every mouse event reach scrcpy below.
            if not self.user32.SetLayeredWindowAttributes(
                wintypes.HWND(hwnd), 0x00030201, 0, 0x00000001
            ):
                raise OSError(ctypes.get_last_error(), "Не удалось включить прозрачность слоя отладки")
            self.user32.ShowWindow(wintypes.HWND(hwnd), 0)
            self.debug_overlay_window = overlay
            self.debug_overlay_canvas = canvas
            self.debug_overlay_hwnd = hwnd
            self.debug_overlay_rect = None
            self.debug_overlay_source_offset = (0, 0)
            self.debug_overlay_visible = False
            self.debug_overlay_draw_key = None
            return True
        except (OSError, tk.TclError, AttributeError, TypeError, ValueError) as error:
            if overlay is not None:
                try:
                    overlay.destroy()
                except tk.TclError:
                    pass
            self.debug_overlay_window = None
            self.debug_overlay_canvas = None
            self.debug_overlay_hwnd = None
            self.debug_overlay_visible = False
            self.debug_overlay_draw_key = None
            self._queue_log(f"Не удалось создать прозрачный слой live-отладки: {error}")
            return False

    def _hide_debug_overlay(self) -> None:
        if not self.debug_overlay_visible or self.user32 is None or self.debug_overlay_hwnd is None:
            return
        try:
            self.user32.ShowWindow(wintypes.HWND(self.debug_overlay_hwnd), 0)
        except (OSError, ValueError):
            pass
        self.debug_overlay_visible = False
        self.debug_overlay_rect = None

    def _sync_debug_overlay(self) -> None:
        if (
            not (self.debug_live or self.protected_zones or self.zone_selection_active)
            or self.scrcpy_hwnd is None
            or self.user32 is None
            or not self.phone.scrcpy_running
            or not self.phone.scrcpy_video_ready
        ):
            self._hide_debug_overlay()
            return
        if not self._ensure_debug_overlay() or self.debug_overlay_hwnd is None:
            return
        try:
            foreground = int(self.user32.GetAncestor(self.user32.GetForegroundWindow(), 2) or 0)
            app_window = int(self.user32.GetAncestor(wintypes.HWND(self.root.winfo_id()), 2) or 0)
            scrcpy_window = int(self.user32.GetAncestor(wintypes.HWND(self.scrcpy_hwnd), 2) or 0)
            app_visible = self.root.state() != "iconic" and self.screen_label.winfo_ismapped()
            if not app_visible or foreground not in {app_window, scrcpy_window}:
                self._hide_debug_overlay()
                return
            scrcpy_hwnd = wintypes.HWND(self.scrcpy_hwnd)
            scrcpy_client = wintypes.RECT()
            scrcpy_origin = wintypes.POINT(0, 0)
            if not self.user32.GetClientRect(scrcpy_hwnd, ctypes.byref(scrcpy_client)):
                return
            if not self.user32.ClientToScreen(scrcpy_hwnd, ctypes.byref(scrcpy_origin)):
                return
            panel_hwnd = wintypes.HWND(self.screen_label.winfo_id())
            panel_client = wintypes.RECT()
            panel_origin = wintypes.POINT(0, 0)
            if not self.user32.GetClientRect(panel_hwnd, ctypes.byref(panel_client)):
                return
            if not self.user32.ClientToScreen(panel_hwnd, ctypes.byref(panel_origin)):
                return

            # Confine the native overlay to the phone preview panel. This keeps
            # a stale or oversized scrcpy rectangle from covering AndAI controls.
            scrcpy_left = scrcpy_origin.x
            scrcpy_top = scrcpy_origin.y
            scrcpy_right = scrcpy_left + max(0, scrcpy_client.right - scrcpy_client.left)
            scrcpy_bottom = scrcpy_top + max(0, scrcpy_client.bottom - scrcpy_client.top)
            self.debug_overlay_source_size = (scrcpy_right - scrcpy_left, scrcpy_bottom - scrcpy_top)
            panel_left, panel_top = panel_origin.x, panel_origin.y
            panel_right = panel_left + max(0, panel_client.right - panel_client.left)
            panel_bottom = panel_top + max(0, panel_client.bottom - panel_client.top)
            x, y = max(scrcpy_left, panel_left), max(scrcpy_top, panel_top)
            right, bottom = min(scrcpy_right, panel_right), min(scrcpy_bottom, panel_bottom)
            width, height = right - x, bottom - y
            if width < 2 or height < 2:
                self._hide_debug_overlay()
                return
            self.debug_overlay_source_offset = (x - scrcpy_left, y - scrcpy_top)
            new_rect = (x, y, width, height)
            if new_rect != self.debug_overlay_rect:
                positioned = self.user32.SetWindowPos(
                    wintypes.HWND(self.debug_overlay_hwnd),
                    wintypes.HWND(-1),
                    x, y, width, height,
                    0x0010 | 0x0040,  # SWP_NOACTIVATE | SWP_SHOWWINDOW
                )
                if not positioned:
                    raise OSError(ctypes.get_last_error(), "Не удалось совместить слой отладки с окном scrcpy")
                self.debug_overlay_rect = new_rect
            self.debug_overlay_error = ""
            self.debug_overlay_visible = True
            self._draw_debug_overlay(width, height)
        except (OSError, tk.TclError, ValueError) as error:
            message = f"Не удалось обновить слой live-отладки: {error}"
            if message != self.debug_overlay_error:
                self.debug_overlay_error = message
                self._queue_log(message)
            self._hide_debug_overlay()

    def _draw_debug_overlay(self, width: int, height: int) -> None:
        canvas = self.debug_overlay_canvas
        if canvas is None or not (self.debug_live or self.protected_zones or self.zone_selection_active):
            return
        frame_width, frame_height = self.frame_size
        annotations = self.debug_annotations
        draw_key = (
            width,
            height,
            self.debug_overlay_source_offset,
            self.debug_overlay_source_size,
            frame_width,
            frame_height,
            self.debug_live,
            tuple(self.protected_zones),
            self.zone_selection_start,
            self.zone_selection_current,
            tuple((item.get("kind"), tuple(item.get("bounds", ())), item.get("label")) for item in annotations),
        )
        if draw_key == self.debug_overlay_draw_key:
            return
        self.debug_overlay_draw_key = draw_key
        canvas.configure(width=width, height=height)
        canvas.delete("debug")
        offset_x, offset_y = self.debug_overlay_source_offset
        source_width, source_height = self.debug_overlay_source_size
        if source_width > 0 and source_height > 0:
            for number, (left, top, right, bottom) in enumerate(self.protected_zones, start=1):
                x1 = round(left * source_width) - offset_x
                y1 = round(top * source_height) - offset_y
                x2 = round(right * source_width) - offset_x
                y2 = round(bottom * source_height) - offset_y
                canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#ff3030", width=3, tags="debug",
                )
                text_id = canvas.create_text(
                    x1 + 4, y1 + 4, anchor="nw", text=f"ИИ ЗАПРЕЩЕНО · {number}",
                    fill="#ffffff", font=("Segoe UI", 9, "bold"), tags="debug",
                )
                text_bounds = canvas.bbox(text_id)
                if text_bounds:
                    bg_id = canvas.create_rectangle(
                        *text_bounds, fill="#d02020", outline="#d02020", tags="debug",
                    )
                    canvas.tag_lower(bg_id, text_id)
            if self.zone_selection_start is not None and self.zone_selection_current is not None:
                canvas.create_rectangle(
                    *self.zone_selection_start, *self.zone_selection_current,
                    outline="#ff3030", width=2, dash=(5, 3), tags="debug",
                )

        if not self.debug_live or frame_width <= 0 or frame_height <= 0:
            return
        scale_x, scale_y = width / frame_width, height / frame_height
        ui_items = [item for item in annotations if item.get("kind") == "UI"][:36]
        ocr_items = [item for item in annotations if item.get("kind") == "OCR"][:36]
        visible_items = [("OCR", item) for item in ocr_items] + [("UI", item) for item in ui_items]
        canvas.create_rectangle(5, 5, 112, 25, fill="#17202b", outline="#17202b", tags="debug")
        canvas.create_text(11, 15, anchor="w", text="OCR", fill="#ff9d2e", font=("Segoe UI", 8, "bold"), tags="debug")
        canvas.create_text(57, 15, anchor="w", text="UI", fill="#00d5b4", font=("Segoe UI", 8, "bold"), tags="debug")
        for number, (kind, item) in enumerate(visible_items, start=1):
            x1, y1, x2, y2 = item.get("bounds", (0, 0, 0, 0))
            x1, x2 = round(x1 * scale_x) - offset_x, round(x2 * scale_x) - offset_x
            y1, y2 = round(y1 * scale_y) - offset_y, round(y2 * scale_y) - offset_y
            color = "#00d5b4" if kind == "UI" else "#ff9d2e"
            canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="debug")
            label = str(item.get("label", "")).replace("\n", " ").strip()
            caption = f"{number}. {label[:72]}" if label else str(number)
            text_id = canvas.create_text(
                x1 + 2, max(1, y1 - 3), anchor="sw", text=caption,
                fill="#ffffff", font=("Segoe UI", 8, "bold"), tags="debug",
            )
            text_bounds = canvas.bbox(text_id)
            if text_bounds:
                bg_id = canvas.create_rectangle(*text_bounds, fill=color, outline=color, tags="debug")
                canvas.tag_lower(bg_id, text_id)

    def _schedule_scrcpy_overlay_sync(self, _event: object = None) -> None:
        if self.scrcpy_hwnd is None or self.scrcpy_sync_after_id is not None:
            return
        try:
            self.scrcpy_sync_after_id = self.root.after(16, self._run_scheduled_scrcpy_sync)
        except tk.TclError:
            self.scrcpy_sync_after_id = None

    def _run_scheduled_scrcpy_sync(self) -> None:
        self.scrcpy_sync_after_id = None
        self._sync_scrcpy_overlay()

    def _sync_scrcpy_overlay(self) -> None:
        window = self.scrcpy_hwnd
        user32 = self.user32
        if (
            window is None
            or user32 is None
            or not self.phone.scrcpy_running
            or not self.phone.scrcpy_video_ready
        ):
            return
        hwnd = wintypes.HWND(window)
        try:
            foreground = int(user32.GetAncestor(user32.GetForegroundWindow(), 2) or 0)  # GA_ROOT
            app_window = int(user32.GetAncestor(wintypes.HWND(self.root.winfo_id()), 2) or 0)
            own_windows = {app_window, window}
            app_is_active = foreground in own_windows
            app_visible = self.root.state() != "iconic" and self.screen_label.winfo_ismapped()
            if not app_visible:
                if not self.scrcpy_overlay_hidden:
                    user32.ShowWindow(hwnd, 0)  # SW_HIDE
                    self.scrcpy_overlay_hidden = True
                self.scrcpy_last_app_active = None
                return

            panel_hwnd = wintypes.HWND(self.screen_label.winfo_id())
            panel_rect = wintypes.RECT()
            panel_origin = wintypes.POINT(0, 0)
            if not user32.GetClientRect(panel_hwnd, ctypes.byref(panel_rect)):
                return
            if not user32.ClientToScreen(panel_hwnd, ctypes.byref(panel_origin)):
                return
            x = panel_origin.x
            y = panel_origin.y
            max_width = panel_rect.right - panel_rect.left
            max_height = panel_rect.bottom - panel_rect.top
            if max_width < 2 or max_height < 2:
                return

            current = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(current)):
                return
            current_width = max(1, current.right - current.left)
            current_height = max(1, current.bottom - current.top)
            if self.scrcpy_aspect_ratio is None:
                self.scrcpy_aspect_ratio = current_width / current_height
            aspect = max(0.1, self.scrcpy_aspect_ratio)
            if aspect > max_width / max_height:
                width = max_width
                height = max(1, round(width / aspect))
            else:
                height = max_height
                width = max(1, round(height * aspect))
            x += (max_width - width) // 2
            y += (max_height - height) // 2

            swp_noactivate = 0x0010
            swp_showwindow = 0x0040
            swp_nomove = 0x0002
            swp_nosize = 0x0001
            swp_nozorder = 0x0004
            geometry_changed = (current.left, current.top, current_width, current_height) != (x, y, width, height)
            zorder_changed = self.scrcpy_last_app_active is None or app_is_active != self.scrcpy_last_app_active
            if geometry_changed or self.scrcpy_overlay_hidden or zorder_changed:
                flags = swp_noactivate | swp_showwindow
                if not app_is_active:
                    # Don't raise scrcpy over another application when AndAI loses focus.
                    flags |= swp_nozorder
                if geometry_changed or self.scrcpy_overlay_hidden:
                    user32.SetWindowPos(hwnd, wintypes.HWND(0), x, y, width, height, flags)
                else:
                    user32.SetWindowPos(
                        hwnd, wintypes.HWND(0), 0, 0, 0, 0,
                        flags | swp_nomove | swp_nosize,
                    )
            self.scrcpy_last_app_active = app_is_active
            self.scrcpy_overlay_hidden = False
        except (OSError, tk.TclError, ValueError):
            return

    def _detach_scrcpy_window(self) -> None:
        self._hide_debug_overlay()
        self.scrcpy_hwnd = None
        self.scrcpy_overlay_hidden = True
        self.scrcpy_last_app_active = None
        self.scrcpy_aspect_ratio = None
        self.last_preview_capture_error = ""
        self.preview_stop_event.set()
        with self.preview_lock:
            self.preview_image = None
            self.preview_sequence += 1
        self.frame_size = (0, 0)
        self.photo = None
        if not self.screen_label.winfo_manager():
            self.screen_label.pack(expand=True, fill="both")
        self.screen_label.configure(image="", text="Ожидание scrcpy…")

    def _start_scrcpy_preview(self, window: int) -> None:
        self.preview_stop_event.set()
        stop_event = threading.Event()
        self.preview_stop_event = stop_event
        threading.Thread(target=self._scrcpy_preview_loop, args=(window, stop_event), daemon=True).start()

    def _scrcpy_preview_loop(self, window: int, stop_event: threading.Event) -> None:
        while not self.stop_event.is_set() and not stop_event.is_set():
            if self.scrcpy_hwnd != window or not self.phone.scrcpy_running:
                break
            debug_active = self.debug_live
            if not (debug_active or self.learning_active):
                stop_event.wait(0.1)
                continue
            try:
                with self.capture_lock:
                    image = ImageGrab.grab(window=window).convert("RGB")
                if debug_active and self.debug_live:
                    self.frame_size = image.size
                    if self.last_preview_capture_error:
                        self.last_preview_capture_error = ""
                        self.events.put(("preview_capture_recovered", window))
                    stop_event.wait(0.25)
                    continue
                if stop_event.is_set() or self.scrcpy_hwnd != window:
                    break
                with self.preview_lock:
                    self.preview_image = image
                    self.preview_sequence += 1
                if self.last_preview_capture_error:
                    self.last_preview_capture_error = ""
                    self.events.put(("preview_capture_recovered", window))
            except Exception as error:
                if stop_event.is_set() or self.scrcpy_hwnd != window:
                    break
                message = f"Не удалось получить изображение окна scrcpy: {error}"
                if message != self.last_preview_capture_error:
                    self.last_preview_capture_error = message
                    self._queue_log(message)
                    self.events.put(("preview_capture_error", window, message))
            stop_event.wait(0.04)

    def _show_scrcpy_preview(self, source: Image.Image) -> None:
        image = source.copy()
        self.frame_size = image.size
        image.thumbnail((650, 760), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.screen_label.configure(image=self.photo, text="")
        if self.debug_live:
            ui_count = sum(item["kind"] == "UI" for item in self.debug_annotations)
            ocr_count = sum(item["kind"] == "OCR" for item in self.debug_annotations)
            brief = "; ".join(item["label"] for item in self.debug_annotations if item["kind"] == "OCR")[:180]
            self.screen_info.configure(
                text=f"LIVE · scrcpy · UI {ui_count} · OCR {ocr_count}" + (f" · {brief}" if brief else "")
            )
        else:
            self.screen_info.configure(text="Видеопоток scrcpy • зеркало в окне AndAI")
        self.last_screen_message = ""
        self.status_var.set(f"scrcpy подключен — экран {self.frame_size[0]}×{self.frame_size[1]}")

    def _screen_loop(self) -> None:
        last_message = ""
        scrcpy_attached = False
        previous_serial: str | None = None
        while not self.stop_event.is_set():
            serial = self.phone.serial
            state = self.phone.state
            if serial != previous_serial:
                previous_serial = serial
                self.scrcpy_failed_serial = None

            if serial and state == "device" and os.name == "nt" and self.phone.scrcpy:
                if scrcpy_attached and self.phone.scrcpy_running and self.phone.scrcpy_serial == serial:
                    self.stop_event.wait(0.05)
                    continue
                if scrcpy_attached:
                    self.events.put(("scrcpy_stopped",))
                    scrcpy_attached = False
                    if not self.phone.scrcpy_running and self.phone.scrcpy_serial == serial:
                        self.scrcpy_failed_serial = serial
                        details = self.phone.scrcpy_exit_diagnostics()
                        message = "Процесс scrcpy завершился."
                        if details:
                            message += f" {details}"
                        message += " Нажмите «Применить путь к scrcpy», чтобы запустить его снова."
                        self.events.put(("screen_message", message))
                        self.events.put(("log", message))
                        last_message = message
                    self.phone.stop_scrcpy()
                if self.scrcpy_failed_serial != serial:
                    try:
                        process_id, title = self.phone.start_scrcpy()
                        window = self._wait_for_scrcpy_window(process_id, title)
                        if window is None:
                            raise RuntimeError("scrcpy не создал окно видеопотока")
                        self.events.put(("scrcpy_started", window))
                        scrcpy_attached = True
                        last_message = ""
                        continue
                    except Exception as error:
                        self.phone.stop_scrcpy()
                        self.events.put(("scrcpy_stopped",))
                        message = f"Не удалось запустить scrcpy: {error}"
                        self.events.put(("screen_message", message))
                        self.events.put(("log", message))
                        last_message = message
                        self.scrcpy_failed_serial = serial
                elif last_message and self.last_screen_message != last_message:
                    self.events.put(("screen_message", last_message))
            elif serial and state in ("fastboot", "recovery", "rebooting"):
                self.phone.stop_scrcpy()
                if scrcpy_attached:
                    self.events.put(("scrcpy_stopped",))
                    scrcpy_attached = False
                message = "scrcpy не передаёт изображение в Fastboot, Recovery или во время перезагрузки."
                if message != last_message:
                    self.events.put(("screen_message", message))
                    last_message = message
            elif serial and state == "device" and os.name != "nt":
                self.phone.stop_scrcpy()
                if scrcpy_attached:
                    self.events.put(("scrcpy_stopped",))
                    scrcpy_attached = False
                message = "Встроенный видеопоток scrcpy в этой программе поддержан только в Windows."
                if message != last_message:
                    self.events.put(("screen_message", message))
                    last_message = message
            elif serial and state == "device" and not self.phone.scrcpy:
                self.phone.stop_scrcpy()
                if scrcpy_attached:
                    self.events.put(("scrcpy_stopped",))
                    scrcpy_attached = False
                message = "Укажите путь к scrcpy.exe: ADB не используется как источник изображения."
                if message != last_message:
                    self.events.put(("screen_message", message))
                    last_message = message
            else:
                self.phone.stop_scrcpy()
                if scrcpy_attached:
                    self.events.put(("scrcpy_stopped",))
                    scrcpy_attached = False
                message = "Нет подключённого и авторизованного Android устройства для scrcpy. Проверьте USB debugging."
                if message != last_message:
                    self.events.put(("screen_message", message))
                    last_message = message
            self.stop_event.wait(0.1)

    def run_ai(self, task_override: str | None = None, force_curiosity: bool = False) -> None:
        if self.ai_busy:
            return
        if self.learning_active:
            messagebox.showinfo("Самообучение", "Сначала остановите запись демонстрации.")
            return
        task = task_override if task_override is not None else self.task_box.get("1.0", "end").strip()
        if not task or task == TASK_PLACEHOLDER:
            messagebox.showinfo("Сообщение для ИИ", "Введите сообщение или попросите ИИ выполнить действие.")
            return
        key, model = self.api_key_var.get().strip(), self.model_var.get().strip()
        ollama_url = self.ollama_url_var.get().strip()
        ollama_model = self.ollama_model_var.get().strip()
        torch_brain_path = self.torch_brain_path_var.get().strip()
        mode = self.mode_var.get()
        curiosity = force_curiosity or self.curiosity_var.get()
        if mode in ("api", "ollama", "torch") and (self.scrcpy_hwnd is None or not self.phone.scrcpy_running):
            messagebox.showinfo(
                "Экран недоступен",
                "Для анализа изображения нужен запущенный scrcpy. Проверьте USB-подключение и путь к scrcpy.exe.",
            )
            return
        if mode == "torch" and not curiosity:
            messagebox.showinfo(
                "Необученная Torch-модель",
                "Случайная модель пока не умеет понимать сообщения и выполнять команды по смыслу. "
                "Включите режим любопытства или нажмите «Исследовать экран».",
            )
            return
        if curiosity and mode == "local":
            messagebox.showinfo("Режим любопытства", "Для исследования выберите Torch / случайная модель, Ollama / Qwen или OpenAI API.")
            return
        if curiosity and (not self.phone.serial or self.phone.state != "device"):
            messagebox.showinfo("Режим любопытства", "Подключите Android-телефон через USB и выберите устройство ADB.")
            return
        autonomous = self.autonomous_var.get() or curiosity
        self._get_action_delay_ms()
        brain_path = self.brain_path_var.get().strip()
        if task_override is None:
            self.task_box.delete("1.0", "end")
            self._refresh_dictionary_preview()
        if task_override is None:
            self._append_chat("Вы", task)
            conversation = self.chat_messages[:-1][-16:]
        else:
            conversation = self.chat_messages[-16:]
        self.ai_stop_event.clear()
        self.ai_busy = True
        self.ai_curiosity_active = curiosity
        if self.curiosity_button is not None and curiosity:
            self.curiosity_button.configure(
                text="Остановить исследование", style="Danger.TButton",
                image=self.material_icons.get(("white", "stop")), command=self.stop_ai, state="normal",
            )
        elif self.curiosity_button is not None:
            self.curiosity_button.configure(state="disabled")
        self.ai_button.configure(
            text="Остановить исследование" if curiosity else "Остановить ИИ",
            style="Danger.TButton", image=self.material_icons.get(("white", "stop")), compound="left",
            command=self.stop_ai, state="normal",
        )
        matches = dictionary_matches(task)
        vocabulary_note = ", ".join(matches) if matches else "совпадений по словарю нет"
        self._queue_log(f"Статус ИИ: получил задачу «{task}»")
        self._queue_log(f"Словарь RU/EN: {vocabulary_note}")
        self._append_log(
            "Локальный мозг выполняет задачу…" if mode == "local" else
            "Локальная Ollama-модель отвечает и анализирует экран…" if mode == "ollama" else
            "Необученная Torch-модель исследует экран…" if mode == "torch" else
            "ИИ анализирует текущий экран и выполняет задачу…"
        )
        if curiosity:
            self._queue_log("Исследование запущено: одно действие за цикл, до нажатия «Остановить исследование».")

        def work() -> None:
            try:
                label = (
                    f"Ollama/{ollama_model}" if mode == "ollama" else
                    f"Torch/{Path(torch_brain_path).name}" if mode == "torch" else
                    "Локальный мозг" if mode == "local" else "ИИ"
                )
                last_action_at: list[float | None] = [None]
                pending_action_description: list[str | None] = [None]
                action_attempts = [0]

                def run_once(
                    pass_task: str,
                    screenshot: Callable[[], bytes | None],
                    max_steps: int,
                ) -> str:
                    def execute_tool(name: str, arguments: dict[str, Any]) -> str:
                        if self.ai_stop_event.is_set():
                            return "Действие отменено пользователем."
                        if last_action_at[0] is not None:
                            remaining = self.action_delay_ms / 1000 - (time.monotonic() - last_action_at[0])
                            if remaining > 0 and self.ai_stop_event.wait(remaining):
                                self._queue_internal_log("Ожидание действия прервано кнопкой «Стоп».")
                                return "Действие отменено пользователем."
                        if self.ai_stop_event.is_set():
                            return "Действие отменено пользователем."
                        description = self._describe_ai_action(name, arguments)
                        try:
                            result = self.phone.run_ai_tool(
                                name, arguments, allow_autonomous=autonomous
                            )
                        except Exception as error:
                            self._queue_internal_log(f"Не получилось: {description} — {error}")
                            action_attempts[0] += 1
                            last_action_at[0] = time.monotonic()
                            raise
                        action_attempts[0] += 1
                        last_action_at[0] = time.monotonic()
                        pending_action_description[0] = description
                        self._queue_internal_log(f"Действие ИИ: {description}.")
                        return result

                    if mode == "local":
                        if not brain_path:
                            raise ValueError("Укажите JSON файл памяти локального мозга")
                        brain = LocalBrain(brain_path)
                        return brain.run_task(
                            pass_task, execute_tool, self._queue_log,
                            allow_autonomous=autonomous,
                        )
                    if mode == "ollama":
                        agent = OllamaVisionAgent(ollama_url, ollama_model)
                        return agent.run_task(
                            task=pass_task, screenshot=screenshot, execute_tool=execute_tool,
                            report=self._queue_log, autonomous=autonomous, curiosity=curiosity,
                            conversation=conversation, should_stop=self.ai_stop_event.is_set,
                            max_steps=max_steps,
                        )
                    if mode == "torch":
                        if self.torch_agent is None or self.torch_agent.brain_path != torch_brain_path:
                            self.torch_agent = TorchVisionAgent(torch_brain_path)
                        return self.torch_agent.run_task(
                            task=pass_task, screenshot=screenshot, execute_tool=execute_tool,
                            report=self._queue_internal_log, autonomous=autonomous, curiosity=curiosity,
                            conversation=conversation, should_stop=self.ai_stop_event.is_set,
                            screen_nodes=self.phone.ui_hierarchy,
                            max_steps=max_steps,
                        )
                    agent = OpenAIVisionAgent(key, model)
                    return agent.run_task(
                        task=pass_task, screenshot=screenshot, execute_tool=execute_tool,
                        report=self._queue_log, autonomous=autonomous, curiosity=curiosity,
                        conversation=conversation, should_stop=self.ai_stop_event.is_set,
                        max_steps=max_steps,
                    )

                if curiosity:
                    reported_errors: set[str] = set()
                    previous_observation: str | None = None
                    passes = 0
                    while not self.ai_stop_event.is_set():
                        frame = self._capture_for_ai()
                        if frame is None:
                            if "Кадр scrcpy недоступен; ожидание подключения" not in reported_errors:
                                self._queue_log("Исследование: кадр scrcpy недоступен; жду восстановления трансляции.")
                                reported_errors.add("Кадр scrcpy недоступен; ожидание подключения")
                            if self.ai_stop_event.wait(1.0):
                                break
                            continue

                        observation, _source, errors = self._curiosity_screen_text(frame)
                        dictionary_terms = dictionary_matches(observation) if observation else []
                        action_note = pending_action_description[0]
                        try:
                            memory = self._remember_screen(frame, observation, dictionary_terms, action_note)
                        except Exception as error:
                            memory = {"known": False, "visits": 0, "error": str(error)}
                        if observation != previous_observation or action_note or not memory.get("known"):
                            self._queue_thought(
                                self._screen_thought(
                                    observation, dictionary_terms, previous_observation, action_note, memory
                                )
                            )
                        pending_action_description[0] = None
                        previous_observation = observation
                        for error in errors:
                            if error not in reported_errors:
                                reported_errors.add(error)
                                self._queue_log(f"Исследование · распознавание: {error}")

                        pass_task = (
                            f"{task}\nКраткое распознавание текущего экрана: {observation[:420] or 'текст не распознан'}. "
                            f"Совпадения RU/EN словаря: {', '.join(dictionary_terms[:12]) or 'нет'}. "
                            f"Запись долговременной памяти экрана: {'уже встречалась' if memory.get('known') else 'новый экран'}; "
                            f"посещений: {memory.get('visits', 1)}; "
                            f"последнее действие на похожем экране: {memory.get('last_action') or 'нет данных'}. "
                            "Выполняй любые нужные действия, включая настройки, ввод текста, приложения, ADB shell и Fastboot. "
                            "Не инициируй и не подтверждай вызовы экстренным службам."
                        )
                        first_frame: list[bytes | None] = [frame]

                        def pass_screenshot() -> bytes | None:
                            if first_frame[0] is not None:
                                captured, first_frame[0] = first_frame[0], None
                                return captured
                            return self._capture_for_ai()

                        attempts_before = action_attempts[0]
                        result = run_once(pass_task, pass_screenshot, 1)
                        passes += 1
                        self.events.put(("status", f"Исследование · цикл {passes}"))
                        if mode == "torch":
                            self._queue_internal_log(f"{label} · цикл {passes} завершён.")
                        else:
                            self._queue_internal_log(f"{label} · исследование, цикл {passes} завершён.")
                            if action_attempts[0] == attempts_before and result.strip():
                                self._queue_internal_log(f"Ответ модели: {result[:300]}")
                        if action_attempts[0] == attempts_before and self.ai_stop_event.wait(self.action_delay_ms / 1000):
                            break

                    result = f"Исследование остановлено пользователем после {passes} циклов."
                else:
                    result = run_once(task, self._capture_for_ai, 12)
                if curiosity:
                    self._queue_internal_log(f"{label}: сводка — {result}")
                else:
                    self._queue_log(f"{label}: сводка — {result}")
                self.events.put(("ai_reply", result))
                self.events.put(("status", result))
            except Exception as error:
                self._queue_log(f"ИИ: ошибка — {error}")
                self.events.put(("ai_reply", f"Ошибка: {error}"))
                self.events.put(("status", str(error)))
            finally:
                self.events.put(("ai_done",))
        threading.Thread(target=work, daemon=True).start()

    def stop_ai(self) -> None:
        self.ai_stop_event.set()
        self.ai_button.configure(text="Останавливается…", state="disabled")
        if self.curiosity_button is not None and self.ai_curiosity_active:
            self.curiosity_button.configure(text="Остановка…", state="disabled")
        self._queue_internal_log("Запрошена остановка ИИ; текущий сетевой запрос завершится перед остановкой.")

    def _capture_for_ai(self) -> bytes | None:
        """Capture pixels from the running scrcpy window; never request an ADB screenshot."""
        if os.name != "nt" or self.scrcpy_hwnd is None or not self.phone.scrcpy_running:
            return None
        try:
            with self.capture_lock:
                image = ImageGrab.grab(window=self.scrcpy_hwnd).convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                png = buffer.getvalue()
            self.last_capture_error = ""
            return png
        except Exception as error:
            message = f"Не удалось получить кадр из scrcpy: {error}"
            if message != self.last_capture_error:
                self.last_capture_error = message
                self._queue_log(message)
            return None

    def _curiosity_screen_text(self, png: bytes) -> tuple[str, str, list[str]]:
        _annotations, recognized_text, errors = self._recognize_debug_frame(png, include_ui=False)
        if recognized_text.strip():
            return " ".join(recognized_text.split()), "OCR", errors
        try:
            root = ET.fromstring(self.phone.ui_hierarchy())
            visible_text: list[str] = []
            seen: set[str] = set()
            for node in root.iter("node"):
                if node.attrib.get("visible-to-user", "true") != "true":
                    continue
                text = " ".join((node.attrib.get("text", ""), node.attrib.get("content-desc", ""))).strip()
                normalized = " ".join(text.casefold().split())
                if text and normalized not in seen:
                    visible_text.append(text)
                    seen.add(normalized)
                if len(visible_text) >= 80:
                    break
            if visible_text:
                return " ".join(visible_text), "Android UI", errors
        except Exception as error:
            errors.append(f"Дерево Android UI: {error}")
        return "", "экран", errors

    def _guard_ai_action(self, name: str, arguments: dict[str, Any]) -> None:
        screen_actions: list[tuple[str, dict[str, Any]]] = []
        if name in {"tap", "hold_screen", "swipe", "press_key", "type_text"}:
            screen_actions.append((name, arguments))
        elif name == "run_adb_shell":
            command = str(arguments.get("command", ""))
            PhoneController._reject_emergency_call_command(command)
            for match in re.finditer(
                r"(?<![\w.-])input\s+(tap|swipe|text|keyevent)\s+([^;&|\n]+)",
                command,
                flags=re.IGNORECASE,
            ):
                operation, raw_args = match.groups()
                try:
                    values = shlex.split(raw_args)
                except ValueError:
                    values = raw_args.split()
                operation = operation.casefold()
                if operation == "tap" and len(values) >= 2:
                    screen_actions.append(("tap", {"x": values[0], "y": values[1]}))
                elif operation == "swipe" and len(values) >= 4:
                    screen_actions.append(("swipe", {
                        "x1": values[0], "y1": values[1], "x2": values[2], "y2": values[3],
                    }))
                elif operation == "text" and values:
                    screen_actions.append(("type_text", {"text": " ".join(values)}))
                elif operation == "keyevent" and values:
                    screen_actions.append(("press_key", {"key": values[0].casefold()}))
        if not screen_actions:
            return
        frame = self._capture_for_ai()
        if frame is None:
            raise RuntimeError("Действие с экраном остановлено: сейчас нельзя проверить кадр scrcpy на вызов экстренной службы.")
        annotations, screen_text, _errors = self._recognize_debug_frame(frame, include_ui=False)
        normalized = screen_text.casefold()
        emergency_labels = ("emergency", "экстр", "sos", "notruf", "urgence", "emergencia")
        phone_labels = ("phone", "dial", "call", "телефон", "набор", "вызов", "позвон")
        has_emergency_label = any(label in normalized for label in emergency_labels)
        has_phone_context = any(label in normalized for label in phone_labels) or has_emergency_label
        emergency_numbers = PhoneController.EMERGENCY_NUMBERS
        has_emergency_number = any(
            re.search(rf"(?<!\d){re.escape(number)}(?!\d)", normalized)
            for number in emergency_numbers
        )

        bounds_to_block = [
            item["bounds"] for item in annotations
            if item.get("kind") == "OCR"
            and any(label in str(item.get("label", "")).casefold() for label in emergency_labels)
        ]
        if has_emergency_number:
            bounds_to_block.extend(
                item["bounds"] for item in annotations
                if item.get("kind") == "OCR"
                and any(label in str(item.get("label", "")).casefold() for label in phone_labels)
            )

        for action, action_arguments in screen_actions:
            if action == "type_text":
                typed_digits = re.sub(r"\D", "", str(action_arguments.get("text", "")))
                if typed_digits in emergency_numbers and has_phone_context:
                    raise RuntimeError("Ввод номера экстренной службы заблокирован.")
            if not has_phone_context:
                continue
            if action in {"tap", "hold_screen"}:
                try:
                    x, y = int(action_arguments.get("x", -1)), int(action_arguments.get("y", -1))
                except (TypeError, ValueError):
                    continue
                if any(x1 - 100 <= x <= x2 + 100 and y1 - 70 <= y <= y2 + 70 for x1, y1, x2, y2 in bounds_to_block):
                    raise RuntimeError("Нажатие на экран вызова экстренной службы заблокировано.")
                if has_emergency_number:
                    raise RuntimeError("Касания на экране набора экстренного номера заблокированы.")
            elif action == "swipe" and has_emergency_number:
                raise RuntimeError("Жест на экране набора экстренного номера заблокирован.")
            elif action == "press_key" and has_emergency_number:
                key = str(action_arguments.get("key", "")).casefold()
                if key in {"enter", "call", "5", "66", "keycode_call", "keycode_enter"}:
                    raise RuntimeError("Подтверждение вызова экстренной службы заблокировано.")

    def debug_screen(self) -> None:
        if self.debug_live:
            self._stop_live_debug()
            return
        if self.learning_active:
            messagebox.showinfo("Отладка экрана", "Сначала остановите запись самообучения.")
            return
        if not self.phone.serial or self.phone.state != "device":
            messagebox.showerror("Экран недоступен", "Подключите и выберите Android-телефон с активным ADB.")
            return
        if self.scrcpy_hwnd is None or not self.phone.scrcpy_running:
            messagebox.showerror("Экран недоступен", "Сначала запустите scrcpy и проверьте путь к scrcpy.exe.")
            return

        self.debug_live = True
        self.debug_annotations = []
        self.debug_ui_annotations = []
        self.debug_ocr_annotations = []
        self.debug_wakeup.clear()
        self.debug_ui_wakeup.clear()
        self._refresh_debug_display()
        if self.debug_button is not None:
            self.debug_button.configure(text="Выключить live-отладку")
        self._queue_log("Live-отладка включена: OCR обновляется по кадрам scrcpy, дерево Android читается отдельно.")
        if self.debug_ocr_thread is None or not self.debug_ocr_thread.is_alive():
            self.debug_ocr_thread = threading.Thread(target=self._live_debug_loop, daemon=True)
            self.debug_ocr_thread.start()
        if self.debug_ui_thread is None or not self.debug_ui_thread.is_alive():
            self.debug_ui_thread = threading.Thread(target=self._live_ui_debug_loop, daemon=True)
            self.debug_ui_thread.start()

    def _stop_live_debug(self) -> None:
        self.debug_live = False
        self.debug_wakeup.set()
        self.debug_ui_wakeup.set()
        self.debug_annotations = []
        self.debug_ui_annotations = []
        self.debug_ocr_annotations = []
        self._refresh_debug_display()
        if self.debug_button is not None:
            self.debug_button.configure(text="Включить live-отладку")
        self._hide_debug_overlay()
        self._queue_log("Live-отладка выключена; видеопоток scrcpy продолжает работать.")

    def _live_debug_loop(self) -> None:
        last_text = ""
        last_error = ""
        while self.debug_live and not self.stop_event.is_set():
            try:
                png = self._capture_for_ai()
                if png is None:
                    raise RuntimeError("ожидание кадра из scrcpy")
                annotations, recognized_text, errors = self._recognize_debug_frame(png, include_ui=False)
                self.events.put(("debug_ocr_annotations", annotations))
                if recognized_text and recognized_text != last_text:
                    last_text = recognized_text
                    self._queue_log(f"OCR экрана: {recognized_text[:700]}")
                    matches = dictionary_matches(recognized_text)
                    self._queue_log("Словарь RU/EN: " + (", ".join(matches) if matches else "совпадений нет"))
                error_note = " · ".join(errors)
                if error_note and error_note != last_error:
                    last_error = error_note
                    self._queue_log(f"Live-отладка: {error_note}")
                elif not error_note:
                    last_error = ""
            except Exception as error:
                error_note = str(error)
                if error_note != last_error:
                    last_error = error_note
                    self._queue_log(f"Live-отладка: {error_note}")
            self.debug_wakeup.wait(1.5)
            self.debug_wakeup.clear()

    def _live_ui_debug_loop(self) -> None:
        last_error = ""
        last_summary = ""
        while self.debug_live and not self.stop_event.is_set():
            try:
                frame = self._capture_for_ai()
                if frame is None:
                    raise RuntimeError("ожидание кадра scrcpy для сопоставления UI")
                image = Image.open(io.BytesIO(frame))
                frame_width, frame_height = image.size
                device_width, device_height = self.phone.screen_size()
                if (frame_width > frame_height) != (device_width > device_height):
                    device_width, device_height = device_height, device_width
                scale_x = frame_width / max(1, device_width)
                scale_y = frame_height / max(1, device_height)
                root = ET.fromstring(self.phone.ui_hierarchy())
                annotations: list[dict[str, Any]] = []
                for node in root.iter("node"):
                    bounds_match = re.fullmatch(
                        r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]",
                        node.attrib.get("bounds", ""),
                    )
                    if not bounds_match:
                        continue
                    raw_bounds = tuple(int(value) for value in bounds_match.groups())
                    bounds = (
                        max(0, min(frame_width, round(raw_bounds[0] * scale_x))),
                        max(0, min(frame_height, round(raw_bounds[1] * scale_y))),
                        max(0, min(frame_width, round(raw_bounds[2] * scale_x))),
                        max(0, min(frame_height, round(raw_bounds[3] * scale_y))),
                    )
                    text = node.attrib.get("text", "").strip()
                    description = node.attrib.get("content-desc", "").strip()
                    class_name = node.attrib.get("class", "").rsplit(".", 1)[-1]
                    clickable = node.attrib.get("clickable") == "true"
                    visible = node.attrib.get("visible-to-user", "true") == "true"
                    if not visible or (not text and not description and not clickable):
                        continue
                    label = " | ".join(part for part in (
                        f"текст: {text}" if text else "",
                        f"описание: {description}" if description else "",
                        f"тип: {class_name}" if class_name else "",
                        "нажимаемый" if clickable else "",
                    ) if part)
                    annotations.append({"kind": "UI", "bounds": bounds, "label": label})
                    if len(annotations) >= 120:
                        break
                self.events.put(("debug_ui_annotations", annotations))
                last_error = ""
                summary = f"UI-дерево обновлено: {len(annotations)} элементов"
                if summary != last_summary:
                    last_summary = summary
                    self._queue_log(summary)
            except Exception as error:
                error_note = str(error)
                if error_note != last_error:
                    last_error = error_note
                    self._queue_log(f"Live-отладка UI: {error_note}")
            self.debug_ui_wakeup.wait(4.0)
            self.debug_ui_wakeup.clear()

    def _refresh_debug_display(self) -> None:
        self.debug_annotations = [*self.debug_ui_annotations, *self.debug_ocr_annotations]
        if not self.debug_live:
            self.screen_info.configure(text="")
            if self.debug_overlay_canvas is not None:
                self.debug_overlay_canvas.delete("debug")
            self.debug_overlay_draw_key = None
            self._hide_debug_overlay()
            return
        ui_count = len(self.debug_ui_annotations)
        ocr_count = len(self.debug_ocr_annotations)
        brief = "; ".join(item["label"] for item in self.debug_ocr_annotations)[:180]
        self.screen_info.configure(
            text=f"LIVE · scrcpy · UI {ui_count} · OCR {ocr_count}" + (f" · {brief}" if brief else "")
        )
        if self.debug_overlay_rect is not None:
            self._draw_debug_overlay(self.debug_overlay_rect[2], self.debug_overlay_rect[3])

    def _recognize_debug_frame(
        self, png: bytes, *, include_ui: bool = True
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        annotations: list[dict[str, Any]] = []
        errors: list[str] = []
        if include_ui:
            try:
                root = ET.fromstring(self.phone.ui_hierarchy())
                for node in root.iter("node"):
                    bounds_match = re.fullmatch(
                        r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]",
                        node.attrib.get("bounds", ""),
                    )
                    if not bounds_match:
                        continue
                    bounds = tuple(int(value) for value in bounds_match.groups())
                    text = node.attrib.get("text", "").strip()
                    description = node.attrib.get("content-desc", "").strip()
                    class_name = node.attrib.get("class", "").rsplit(".", 1)[-1]
                    clickable = node.attrib.get("clickable") == "true"
                    visible = node.attrib.get("visible-to-user", "true") == "true"
                    if not visible or (not text and not description and not clickable):
                        continue
                    label = " | ".join(part for part in (
                        f"текст: {text}" if text else "",
                        f"описание: {description}" if description else "",
                        f"тип: {class_name}" if class_name else "",
                        "нажимаемый" if clickable else "",
                    ) if part)
                    annotations.append({"kind": "UI", "bounds": bounds, "label": label})
                    if len(annotations) >= 120:
                        break
            except Exception as error:
                errors.append(f"UI-дерево: {error}")

        texts: list[str] = []
        try:
            import pytesseract

            tesseract = shutil.which("tesseract")
            if not tesseract and os.name == "nt":
                known_paths = (
                    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
                    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
                )
                tesseract = next((str(path) for path in known_paths if path.is_file()), None)
            if not tesseract:
                raise RuntimeError("Не найден tesseract.exe. Установите Tesseract OCR или добавьте его в PATH.")
            pytesseract.pytesseract.tesseract_cmd = tesseract
            tessdata_dir = Path(__file__).resolve().parent / "tessdata"
            missing_languages = [
                language for language in ("eng", "rus")
                if not (tessdata_dir / f"{language}.traineddata").is_file()
            ]
            if missing_languages:
                raise RuntimeError(
                    f"Не найдены языковые данные {', '.join(missing_languages)} в {tessdata_dir}. "
                    "Добавьте файлы .traineddata в папку tessdata проекта."
                )
            config = f'--tessdata-dir "{tessdata_dir.as_posix()}" --psm 11'
            available_languages = pytesseract.get_languages(config=config)
            selected_languages = [language for language in ("rus", "eng") if language in available_languages]
            if not selected_languages:
                raise RuntimeError(f"Tesseract не загрузил языки из папки {tessdata_dir}.")
            image = Image.open(io.BytesIO(png)).convert("RGB")
            scale = min(1.0, 1000.0 / image.width)
            if scale < 1.0:
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.Resampling.BILINEAR,
                )
            data = pytesseract.image_to_data(
                image,
                lang="+".join(selected_languages),
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            for index, raw_text in enumerate(data.get("text", [])):
                value = str(raw_text).strip()
                try:
                    confidence = float(data["conf"][index])
                    x, y = int(data["left"][index]), int(data["top"][index])
                    width, height = int(data["width"][index]), int(data["height"][index])
                except (ValueError, TypeError, IndexError):
                    continue
                if not value or confidence < 35 or width <= 0 or height <= 0:
                    continue
                texts.append(value)
                bounds = tuple(round(part / scale) for part in (x, y, x + width, y + height))
                annotations.append({
                    "kind": "OCR",
                    "bounds": bounds,
                    "label": f"{value} ({confidence:.0f}%)",
                })
                if len(annotations) >= 240:
                    break
        except Exception as error:
            errors.append(f"OCR rus+eng: {error}")

        return annotations, " ".join(texts), errors

    def _drain_events(self) -> None:
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "log":
                self._append_log(str(event[1]))
            elif kind == "trace":
                self._append_trace(str(event[1]))
            elif kind == "thought":
                self._append_thought(str(event[1]))
            elif kind == "status":
                self.status_var.set(str(event[1]))
            elif kind == "ai_reply":
                self._append_chat("ИИ", str(event[1]))
            elif kind == "devices":
                self._set_devices(event[1], event[2])
            elif kind == "screen_message":
                message = str(event[1])
                if message != self.last_screen_message:
                    self.screen_info.configure(text=message)
                    self.last_screen_message = message
                self.status_var.set(message)
                self.screen_label.configure(image="", text=message)
            elif kind == "scrcpy_started":
                try:
                    self._attach_scrcpy_window(int(event[1]))
                except Exception as error:
                    message = f"Не удалось подготовить окно scrcpy: {error}"
                    self._append_log(message)
                    self.events.put(("screen_message", message))
                    self.phone.stop_scrcpy()
                    self.scrcpy_failed_serial = self.phone.serial
                    self._detach_scrcpy_window()
            elif kind == "scrcpy_stopped":
                self._detach_scrcpy_window()
            elif kind == "preview_capture_error" and int(event[1]) == self.scrcpy_hwnd:
                self.screen_info.configure(text="Поток scrcpy запущен, но кадр временно недоступен")
                self.status_var.set("Не удаётся получить кадр scrcpy; повторяю захват")
                if self.photo is None:
                    self.screen_label.configure(image="", text="Повторное получение кадра scrcpy…")
            elif kind == "preview_capture_recovered" and int(event[1]) == self.scrcpy_hwnd:
                if self.photo is None:
                    self.screen_label.configure(image="", text="Получен кадр scrcpy…")
            elif kind in ("debug_ocr_annotations", "debug_ui_annotations", "debug_annotations"):
                if self.debug_live:
                    if kind == "debug_ui_annotations":
                        self.debug_ui_annotations = event[1]
                    elif kind == "debug_ocr_annotations":
                        self.debug_ocr_annotations = event[1]
                    else:
                        self.debug_ui_annotations = [item for item in event[1] if item["kind"] == "UI"]
                        self.debug_ocr_annotations = [item for item in event[1] if item["kind"] == "OCR"]
                    self._refresh_debug_display()
            elif kind == "learning_count":
                if self.learning_active:
                    self.status_var.set(f"Самообучение: записано действий — {event[1]}")
            elif kind == "dictionary_loaded":
                self._refresh_dictionary_preview()
                self._queue_log(f"Словарь: добавлены {event[1]} записей, всего терминов — {event[2]} ({event[3]}).")
                messagebox.showinfo("Словарь загружен", f"Импортировано записей: {event[1]}\nВсего терминов: {event[2]}")
            elif kind == "dictionary_error":
                self._queue_log(f"Словарь: ошибка при {event[2]} — {event[1]}")
                messagebox.showerror("Не удалось загрузить словарь", str(event[1]))
            elif kind == "ai_done":
                self.ai_busy = False
                self.ai_curiosity_active = False
                if self.curiosity_button is not None:
                    self.curiosity_button.configure(
                        text="Исследовать экран", style="Research.TButton",
                        image=self.material_icons.get(("blue", "play_arrow")),
                        command=self.run_curiosity, state="normal",
                    )
                self.ai_button.configure(
                    text="Отправить ИИ / выполнить", style="Primary.TButton",
                    image=self.material_icons.get(("white", "play_arrow")), compound="left",
                    command=self.run_ai, state="normal",
                )
        if not self.stop_event.is_set():
            preview: Image.Image | None = None
            with self.preview_lock:
                if self.preview_sequence != self.displayed_preview_sequence:
                    preview = self.preview_image
                    self.displayed_preview_sequence = self.preview_sequence
            if preview is not None and self.scrcpy_hwnd is not None and self.learning_active:
                self._show_scrcpy_preview(preview)
            self._sync_scrcpy_overlay()
            self._sync_debug_overlay()
            self.root.after(50, self._drain_events)

    def close(self) -> None:
        self.stop_event.set()
        self.ai_stop_event.set()
        self.debug_live = False
        self.debug_wakeup.set()
        self.debug_ui_wakeup.set()
        self.learning_active = False
        self.phone.action_listener = None
        self.phone.stop_scrcpy()
        if self.debug_overlay_window is not None:
            try:
                self.debug_overlay_window.destroy()
            except tk.TclError:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PhoneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
