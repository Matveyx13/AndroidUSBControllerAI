"""ADB/Fastboot helpers for the USB phone controller."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent


def find_binary(name: str) -> str | None:
    """Find adb/fastboot in PATH or a local Android platform-tools folder."""
    suffix = ".exe" if os.name == "nt" else ""
    local = ROOT / "platform-tools" / f"{name}{suffix}"
    return str(local) if local.is_file() else shutil.which(name)


def find_scrcpy(adb_path: str | None = None) -> str | None:
    """Find the scrcpy client beside ADB, in the project, or in PATH."""
    executable = "scrcpy.exe" if os.name == "nt" else "scrcpy"
    candidates = [ROOT / executable]
    if adb_path:
        candidates.insert(0, Path(adb_path).resolve().parent / executable)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("scrcpy")


class PhoneController:
    EMERGENCY_NUMBERS = ("112", "911", "999", "000", "110")
    KEY_CODES = {
        "power": "KEYCODE_POWER",
        "volume_up": "KEYCODE_VOLUME_UP",
        "volume_down": "KEYCODE_VOLUME_DOWN",
        "home": "KEYCODE_HOME",
        "back": "KEYCODE_BACK",
        "recents": "KEYCODE_APP_SWITCH",
        "enter": "KEYCODE_ENTER",
        "menu": "KEYCODE_MENU",
        "wakeup": "KEYCODE_WAKEUP",
        "camera": "KEYCODE_CAMERA",
    }

    def __init__(self) -> None:
        self.adb = find_binary("adb")
        self.fastboot = find_binary("fastboot")
        self.scrcpy = find_scrcpy(self.adb)
        self.serial: str | None = None
        self.state: str | None = None
        self._ui_hierarchy_lock = threading.Lock()
        self._scrcpy_lock = threading.Lock()
        self._scrcpy_process: subprocess.Popen[bytes] | None = None
        self._scrcpy_serial: str | None = None
        self._scrcpy_output_lock = threading.Lock()
        self._scrcpy_output: deque[str] = deque(maxlen=80)
        self._scrcpy_output_done = threading.Event()
        self._scrcpy_output_done.set()
        self._scrcpy_video_ready = threading.Event()
        self.action_listener: Any = None
        self.ai_action_guard: Callable[[str, dict[str, Any]], None] | None = None

    def _notify_action(self, name: str, arguments: dict[str, Any]) -> None:
        """Notify the UI about a completed, replayable action."""
        listener = self.action_listener
        if listener is not None:
            try:
                listener(name, dict(arguments))
            except Exception:
                # Recording must never turn a successful phone action into an error.
                pass

    def set_adb_path(self, path: str) -> str:
        """Use an explicitly selected adb executable, or fall back to PATH."""
        path = path.strip().strip('"')
        if not path:
            self.stop_scrcpy()
            self.adb = find_binary("adb")
            self.fastboot = find_binary("fastboot")
            if not self.scrcpy:
                self.scrcpy = find_scrcpy(self.adb)
            return self.adb or ""

        candidate = Path(path).expanduser()
        if candidate.is_dir():
            candidate = candidate / ("adb.exe" if os.name == "nt" else "adb")
        try:
            candidate = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл adb не найден: {path}") from None
        if not candidate.is_file() or candidate.name.lower() not in ("adb", "adb.exe"):
            raise ValueError("Укажите полный путь к adb.exe (или к файлу adb на другой ОС).")

        self.stop_scrcpy()
        self.adb = str(candidate)
        sibling_fastboot = candidate.with_name("fastboot.exe" if os.name == "nt" else "fastboot")
        self.fastboot = str(sibling_fastboot) if sibling_fastboot.is_file() else shutil.which("fastboot")
        if not self.scrcpy:
            self.scrcpy = find_scrcpy(self.adb)
        return self.adb

    def set_scrcpy_path(self, path: str) -> str:
        path = path.strip().strip('"')
        if not path:
            selected = find_scrcpy(self.adb)
            with self._scrcpy_lock:
                self._stop_scrcpy_locked()
                self.scrcpy = selected
            return self.scrcpy or ""
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            candidate = candidate / ("scrcpy.exe" if os.name == "nt" else "scrcpy")
        try:
            candidate = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл scrcpy не найден: {path}") from None
        if not candidate.is_file() or candidate.name.lower() not in ("scrcpy", "scrcpy.exe"):
            raise ValueError("Укажите путь к scrcpy.exe (или scrcpy на другой ОС).")
        with self._scrcpy_lock:
            self._stop_scrcpy_locked()
            self.scrcpy = str(candidate)
            return self.scrcpy

    @property
    def scrcpy_running(self) -> bool:
        return self._scrcpy_process is not None and self._scrcpy_process.poll() is None

    @property
    def scrcpy_serial(self) -> str | None:
        return self._scrcpy_serial

    @property
    def scrcpy_video_ready(self) -> bool:
        """Whether scrcpy has initialized its video texture for the current process."""
        return self._scrcpy_video_ready.is_set()

    def start_scrcpy(self) -> tuple[int, str]:
        with self._scrcpy_lock:
            if not self.scrcpy:
                raise RuntimeError("scrcpy не найден. Установите scrcpy или укажите путь к scrcpy.exe.")
            if not self.serial or self.state != "device":
                raise RuntimeError("Для scrcpy выберите подключённый Android телефон в режиме device.")
            if self.scrcpy_running and self._scrcpy_serial == self.serial:
                return self._scrcpy_process.pid, f"AndAI scrcpy [{self.serial}]"
            self._stop_scrcpy_locked()
            title = f"AndAI scrcpy [{self.serial}]"
            command = [
                self.scrcpy, "--serial", self.serial, f"--window-title={title}",
                "--window-borderless", "--no-window-aspect-ratio-lock",
                "--no-audio", "--max-size=1280", "--max-fps=30",
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                with self._scrcpy_output_lock:
                    self._scrcpy_output.clear()
                self._scrcpy_video_ready.clear()
                self._scrcpy_output_done.clear()
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            except OSError as error:
                self._scrcpy_output_done.set()
                raise RuntimeError(f"Не удалось запустить scrcpy: {error}") from None
            self._scrcpy_process = process
            self._scrcpy_serial = self.serial
            threading.Thread(target=self._read_scrcpy_output, args=(process,), daemon=True).start()
            return process.pid, title

    def _read_scrcpy_output(self, process: subprocess.Popen[bytes]) -> None:
        try:
            if process.stdout is None:
                return
            for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    with self._scrcpy_output_lock:
                        self._scrcpy_output.append(line[-500:])
                    if self._scrcpy_process is process and "texture" in line.lower():
                        self._scrcpy_video_ready.set()
        except (OSError, ValueError):
            pass
        finally:
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
            self._scrcpy_output_done.set()

    def scrcpy_exit_diagnostics(self) -> str:
        process = self._scrcpy_process
        if process is None or process.poll() is None:
            return ""
        self._scrcpy_output_done.wait(0.3)
        with self._scrcpy_output_lock:
            lines = list(self._scrcpy_output)[-12:]
        details = " | ".join(lines)
        result = f"код завершения {process.returncode}"
        return f"{result}: {details}" if details else result

    def _stop_scrcpy_locked(self) -> None:
        process = self._scrcpy_process
        self._scrcpy_process = None
        self._scrcpy_serial = None
        self._scrcpy_video_ready.clear()
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

    def stop_scrcpy(self) -> None:
        with self._scrcpy_lock:
            self._stop_scrcpy_locked()

    @staticmethod
    def _run(command: list[str], timeout: float = 15) -> subprocess.CompletedProcess[bytes]:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            creationflags=creationflags,
        )

    @staticmethod
    def _decode(raw: bytes) -> str:
        return raw.decode("utf-8", errors="replace").strip()

    def discover(self) -> tuple[list[dict[str, str]], list[str]]:
        devices: list[dict[str, str]] = []
        errors: list[str] = []
        if self.adb:
            result = self._run([self.adb, "devices", "-l"], timeout=8)
            for line in self._decode(result.stdout).splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    devices.append({"serial": parts[0], "state": parts[1], "kind": "ADB"})
            if result.returncode:
                errors.append(self._decode(result.stderr) or "Не удалось запустить ADB")
        else:
            errors.append("adb не найден: добавьте Android platform-tools в PATH")

        if self.fastboot:
            result = self._run([self.fastboot, "devices"], timeout=8)
            for line in self._decode(result.stdout).splitlines():
                parts = line.split()
                if parts:
                    serial = parts[0]
                    existing = next((d for d in devices if d["serial"] == serial), None)
                    if existing:
                        existing.update({"state": "fastboot", "kind": "Fastboot"})
                    else:
                        devices.append({"serial": serial, "state": "fastboot", "kind": "Fastboot"})
            if result.returncode:
                errors.append(self._decode(result.stderr) or "Не удалось запустить Fastboot")
        return devices, errors

    def select(self, serial: str, state: str) -> None:
        self.serial = serial
        self.state = state

    def _adb(self, *args: str, timeout: float = 15) -> subprocess.CompletedProcess[bytes]:
        if not self.adb:
            raise RuntimeError("adb не найден. Установите Android platform-tools и добавьте их в PATH.")
        if not self.serial:
            raise RuntimeError("Сначала выберите телефон в списке устройств.")
        return self._run([self.adb, "-s", self.serial, *args], timeout=timeout)

    def _shell(self, *args: str, timeout: float = 15) -> str:
        # ADB's shell command is parsed on the phone, so quote each argument there.
        command = " ".join(shlex.quote(str(value)) for value in args)
        result = self._adb("shell", command, timeout=timeout)
        return self._checked_shell_output(result, " ".join(map(str, args)))

    @classmethod
    def _checked_shell_output(cls, result: subprocess.CompletedProcess[bytes], command: str) -> str:
        stdout = cls._decode(result.stdout)
        stderr = cls._decode(result.stderr)
        output = "\n".join(part for part in (stdout, stderr) if part)
        lowered = output.casefold()
        input_related = bool(re.search(r"\b(input|tap|swipe|keyevent|motionevent)\b", command.casefold()))
        injection_denied = any(
            marker in lowered
            for marker in (
                "injecting input events requires",
                "injectinputevent",
                "inject_events",
                "permission denial",
                "not allowed to inject",
            )
        )
        android_exception = bool(re.search(
            r"(?i)(?:exception occurred while executing|java\.lang\.[\w.$]*exception|securityexception)",
            output,
        ))
        if input_related and injection_denied:
            hint = (
                "Android заблокировал ADB-ввод (INJECT_EVENTS). На Xiaomi/Redmi/POCO откройте "
                "Настройки → Расширенные настройки → Для разработчиков → «Отладка по USB "
                "(настройки безопасности)» и включите разрешение на выдачу разрешений и "
                "имитацию ввода через USB. После этого перезагрузите телефон, переподключите USB "
                "и повторите действие. Названия пунктов могут отличаться в разных версиях MIUI/HyperOS. "
                "Другие производители могут блокировать ADB-ввод своей политикой; обходить такой запрет "
                "без разрешения ОС нельзя."
            )
            raise RuntimeError(f"{hint}\n\nОтвет Android:\n{output[-3500:]}")
        if result.returncode or android_exception:
            raise RuntimeError(output[-3500:] or "Команда ADB завершилась с ошибкой")
        return stdout

    def _require_adb_device(self) -> None:
        if not self.serial:
            raise RuntimeError("Сначала выберите телефон в списке устройств.")
        if self.state == "fastboot":
            raise RuntimeError("Телефон сейчас в Fastboot. Команды ADB доступны после загрузки Android.")
        if self.state != "device":
            raise RuntimeError("ADB не авторизован. Разблокируйте телефон и подтвердите запрос USB debugging.")

    def ui_hierarchy(self) -> str:
        """Return Android's accessibility hierarchy XML for the current screen."""
        self._require_adb_device()
        remote_path = "/sdcard/andai_window.xml"
        with self._ui_hierarchy_lock:
            try:
                last_error = ""
                for dump_args in (
                    ("shell", "uiautomator", "dump", remote_path),
                    ("shell", "uiautomator", "dump", "--compressed", remote_path),
                ):
                    result = self._adb(*dump_args, timeout=25)
                    if result.returncode:
                        last_error = self._decode(result.stderr) or self._decode(result.stdout)
                        continue
                    xml_result = self._adb("exec-out", "cat", remote_path, timeout=12)
                    xml = xml_result.stdout.lstrip()
                    if xml_result.returncode == 0 and xml.startswith(b"<?xml"):
                        return self._decode(xml_result.stdout)
                    last_error = self._decode(xml_result.stderr) or self._decode(xml_result.stdout)
                raise RuntimeError(last_error or "Android не смог получить дерево элементов экрана после двух попыток")
            finally:
                try:
                    self._adb("shell", "rm", "-f", remote_path, timeout=8)
                except Exception:
                    pass

    def screen_size(self) -> tuple[int, int]:
        self._require_adb_device()
        output = self._shell("wm", "size", timeout=8)
        matches = re.findall(r"(\d+)x(\d+)", output)
        if not matches:
            raise RuntimeError("Не удалось определить размер экрана")
        width, height = matches[-1]
        return int(width), int(height)

    def tap(self, x: int, y: int) -> str:
        self._require_adb_device()
        x, y = max(0, int(x)), max(0, int(y))
        self._shell("input", "tap", str(x), str(y))
        self._notify_action("tap", {"x": x, "y": y})
        return f"Нажатие: {x}, {y}"

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 400) -> str:
        self._require_adb_device()
        duration_ms = min(10000, max(80, int(duration_ms)))
        self._shell(
            "input", "swipe", str(max(0, int(x1))), str(max(0, int(y1))),
            str(max(0, int(x2))), str(max(0, int(y2))), str(duration_ms),
        )
        self._notify_action("swipe", {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "duration_ms": duration_ms})
        return f"Свайп: ({x1}, {y1}) → ({x2}, {y2}), {duration_ms} мс"

    def hold_screen(self, x: int, y: int, duration_ms: int = 1000) -> str:
        self._require_adb_device()
        duration_ms = min(10000, max(100, int(duration_ms)))
        x, y = max(0, int(x)), max(0, int(y))
        self._shell("input", "swipe", str(x), str(y), str(x), str(y), str(duration_ms))
        self._notify_action("hold_screen", {"x": x, "y": y, "duration_ms": duration_ms})
        return f"Удержание экрана: {x}, {y}, {duration_ms} мс"

    def press_key(self, key: str, long_press: bool = False) -> str:
        self._require_adb_device()
        key_code = self.KEY_CODES.get(key)
        if not key_code:
            raise RuntimeError(f"Неизвестная кнопка: {key}")
        args = ["input", "keyevent"]
        if long_press:
            args.append("--longpress")
        args.append(key_code)
        self._shell(*args)
        self._notify_action("press_key", {"key": key, "long_press": bool(long_press)})
        return f"Кнопка {key}{' (долгое нажатие)' if long_press else ''}"

    def type_text(self, text: str) -> str:
        self._require_adb_device()
        # Android's `input text` syntax uses %s in place of a space.
        encoded = str(text).replace(" ", "%s")
        self._shell("input", "text", encoded, timeout=20)
        return "Текст введён"

    def open_app(self, package_name: str) -> str:
        self._require_adb_device()
        package_name = str(package_name).strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", package_name):
            raise RuntimeError("Укажите имя пакета Android, например com.android.settings")
        output = self._shell("monkey", "-p", package_name, "1", timeout=15)
        self._notify_action("open_app", {"package_name": package_name})
        return output or f"Запущено приложение {package_name}"

    def set_network(self, feature: str, enabled: bool) -> str:
        self._require_adb_device()
        services = {"wifi": "wifi", "mobile_data": "data"}
        service = services.get(feature)
        if not service:
            raise RuntimeError("Можно переключить только Wi-Fi или мобильные данные")
        self._shell("svc", service, "enable" if enabled else "disable", timeout=10)
        self._notify_action("set_network", {"feature": feature, "enabled": bool(enabled)})
        label = "Wi-Fi" if feature == "wifi" else "мобильные данные"
        return f"{label}: {'включены' if enabled else 'выключены'} (если разрешено этой версией Android)"

    def set_bluetooth(self, enabled: bool) -> str:
        """Toggle Bluetooth through Android's shell command, with legacy fallback."""
        self._require_adb_device()
        verb = "enable" if enabled else "disable"
        try:
            output = self._shell("cmd", "bluetooth_manager", verb, timeout=20)
        except Exception as modern_error:
            try:
                output = self._shell("svc", "bluetooth", verb, timeout=20)
            except Exception as legacy_error:
                raise RuntimeError(
                    f"Не удалось переключить Bluetooth через cmd или svc: {modern_error}; {legacy_error}"
                ) from None
        self._notify_action("set_bluetooth", {"enabled": bool(enabled)})
        return f"Bluetooth {'включён' if enabled else 'выключен'}" + (f": {output.strip()}" if output.strip() else "")

    def reboot(self, target: str) -> str:
        if not self.serial:
            raise RuntimeError("Сначала выберите телефон в списке устройств.")
        if target == "fastboot":
            self._require_adb_device()
            result = self._adb("reboot", "bootloader", timeout=8)
            if result.returncode:
                raise RuntimeError(self._decode(result.stderr) or "Не удалось перезагрузить в bootloader")
            self.state = "fastboot"
            return "Телефон отправлен в bootloader/Fastboot"
        if target == "recovery":
            self._require_adb_device()
            result = self._adb("reboot", "recovery", timeout=8)
            if result.returncode:
                raise RuntimeError(self._decode(result.stderr) or "Не удалось перезагрузить в Recovery")
            self.state = "recovery"
            return "Телефон отправлен в Recovery"
        if target == "system":
            if self.state == "fastboot":
                if not self.fastboot:
                    raise RuntimeError("fastboot не найден в PATH")
                result = self._run([self.fastboot, "-s", self.serial, "reboot"], timeout=10)
                if result.returncode:
                    raise RuntimeError(self._decode(result.stderr) or "Команда fastboot reboot завершилась с ошибкой")
            else:
                result = self._adb("reboot", timeout=8)
                if result.returncode:
                    raise RuntimeError(self._decode(result.stderr) or "Не удалось перезагрузить Android")
            self.state = "rebooting"
            return "Перезагрузка в Android запрошена"
        raise RuntimeError("Неизвестная цель перезагрузки")

    def run_adb_shell(self, command: str) -> str:
        self._require_adb_device()
        command = str(command).strip()
        if not command:
            raise RuntimeError("Команда ADB shell пустая")
        self._reject_emergency_call_command(command)
        result = self._adb("shell", command, timeout=60)
        output = self._checked_shell_output(result, command)
        return (output or "Команда выполнена")[:6000]

    def run_fastboot(self, command: str) -> str:
        if not self.serial or self.state != "fastboot":
            raise RuntimeError("Для Fastboot команд телефон должен быть выбран в режиме загрузчика")
        if not self.fastboot:
            raise RuntimeError("fastboot не найден. Укажите adb.exe из папки Platform-Tools или добавьте fastboot.exe в PATH")
        try:
            args = shlex.split(str(command), posix=True)
        except ValueError as error:
            raise RuntimeError(f"Неверная команда Fastboot: {error}") from None
        if args and args[0].lower() == "fastboot":
            args.pop(0)
        if not args:
            raise RuntimeError("Команда Fastboot пустая")
        if any(arg in ("-s", "--serial") or arg.startswith("--serial=") for arg in args):
            raise RuntimeError("Нельзя переопределить выбранный serial устройства")
        result = self._run([self.fastboot, "-s", self.serial, *args], timeout=120)
        output = self._decode(result.stdout)
        error = self._decode(result.stderr)
        details = "\n".join(part for part in (output, error) if part)
        if result.returncode:
            raise RuntimeError(details or "Fastboot завершился с ошибкой")
        return (details or "Команда выполнена")[:6000]

    @classmethod
    def _reject_emergency_call_command(cls, command: str) -> None:
        value = str(command).casefold()
        number_present = any(re.search(rf"(?<!\d){re.escape(number)}(?!\d)", value) for number in cls.EMERGENCY_NUMBERS)
        emergency_action = re.search(r"call_emergency|action_call_emergency", value)
        emergency_uri = number_present and re.search(r"tel(?:%3a|:)", value)
        telephony_call = number_present and re.search(
            r"(?:action_call|action_dial|telecom.{0,40}\bcall\b|\bam\s+start\b.{0,100}\b(?:call|dial)\b)",
            value,
        )
        if emergency_action or emergency_uri or telephony_call:
            raise RuntimeError("ИИ заблокировал попытку связаться с экстренной службой.")

    def run_ai_tool(self, name: str, arguments: dict[str, Any], *, allow_autonomous: bool = False) -> str:
        """Execute only the explicit, typed actions offered to the AI."""
        if self.ai_action_guard is not None:
            self.ai_action_guard(name, arguments)
        if name == "run_adb_shell" and allow_autonomous:
            return self.run_adb_shell(arguments["command"])
        if name == "run_fastboot" and allow_autonomous:
            return self.run_fastboot(arguments["command"])
        if name == "tap":
            return self.tap(arguments["x"], arguments["y"])
        if name == "swipe":
            return self.swipe(arguments["x1"], arguments["y1"], arguments["x2"], arguments["y2"], arguments["duration_ms"])
        if name == "hold_screen":
            return self.hold_screen(arguments["x"], arguments["y"], arguments["duration_ms"])
        if name == "press_key":
            return self.press_key(arguments["key"], arguments["long_press"])
        if name == "type_text":
            return self.type_text(arguments["text"])
        if name == "open_app":
            return self.open_app(arguments["package_name"])
        if name == "set_network":
            return self.set_network(arguments["feature"], arguments["enabled"])
        if name == "set_bluetooth":
            return self.set_bluetooth(arguments["enabled"])
        if name == "reboot":
            return self.reboot(arguments["target"])
        raise RuntimeError(f"AI запросил неизвестное действие: {name}")

