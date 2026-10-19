"""Randomly initialized PyTorch policy for cautious Android screen exploration."""

from __future__ import annotations

import io
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

from PIL import Image


_EXCLUDED_LABELS = (
    "emergency call", "emergency services", "экстренн", "служба спас", "notruf", "urgence", "emergencia", "sos",
)
_EMERGENCY_NUMBERS = ("112", "911", "999", "000", "110")


def _screen_tensor(torch: Any, png: bytes, device: Any) -> tuple[Any, tuple[int, int]]:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    size = image.size
    image = image.resize((96, 160), Image.Resampling.BILINEAR)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(160, 96, 3).permute(2, 0, 1).float().div_(255.0)
    return tensor.unsqueeze(0).to(device), size


def _make_policy(torch: Any) -> Any:
    class ScreenPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 96),
                torch.nn.Tanh(),
            )
            self.score = torch.nn.Sequential(
                torch.nn.Linear(105, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1),
            )

        def forward(self, screen: Any, candidates: Any) -> Any:
            visual = self.vision(screen).expand(candidates.shape[0], -1)
            return self.score(torch.cat((visual, candidates), dim=1)).squeeze(1)

    return ScreenPolicy()


class TorchVisionAgent:
    """A small untrained-by-default visual policy, trained from screen changes."""

    def __init__(self, brain_path: str) -> None:
        if not brain_path.strip():
            raise ValueError("Выберите путь к файлу весов Torch-модели (.pt).")
        self.brain_path = str(Path(brain_path).expanduser())
        self._torch: Any = None
        self._model: Any = None
        self._optimizer: Any = None
        self._device: Any = None
        self._baseline = 0.05
        self._lock = threading.Lock()

    def _load(self, report: Callable[[str], None]) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
            except ImportError:
                raise RuntimeError(
                    "Не установлен PyTorch. Установите его командой: "
                    "python -m pip install -r requirements-torch.txt"
                ) from None

            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _make_policy(torch).to(self._device)
            path = Path(self.brain_path)
            if path.is_file():
                try:
                    checkpoint = torch.load(path, map_location=self._device, weights_only=True)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["state_dict"])
                        self._baseline = float(checkpoint.get("reward_baseline", self._baseline))
                    else:
                        model.load_state_dict(checkpoint)
                    report(f"PyTorch: загружены веса из {path.name}.")
                except Exception as error:
                    raise RuntimeError(f"Не удалось загрузить файл весов {path}: {error}") from None
            else:
                report("PyTorch: создана новая модель со случайными весами; она пока ничего не знает.")
            self._model = model
            self._optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def _save(self) -> None:
        path = Path(self.brain_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        self._torch.save(
            {
                "format": 1,
                "state_dict": self._model.state_dict(),
                "reward_baseline": self._baseline,
            },
            temp_path,
        )
        temp_path.replace(path)

    @staticmethod
    def _screen_candidates(
        xml: str | None,
        width: int,
        height: int,
    ) -> list[tuple[dict[str, Any], list[float]]]:
        candidates: list[tuple[dict[str, Any], list[float]]] = []

        def add(action: dict[str, Any], bounds: tuple[int, int, int, int], kind: int) -> None:
            x1, y1, x2, y2 = bounds
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            features = [
                cx / max(width, 1), cy / max(height, 1),
                (x2 - x1) / max(width, 1), (y2 - y1) / max(height, 1),
                float(kind == 0), float(kind == 1), float(kind == 2), float(kind == 3),
                0.0,
            ]
            candidates.append((action, features))

        if xml:
            try:
                root = ET.fromstring(xml)
                seen: set[tuple[int, int]] = set()
                screen_text = " ".join(
                    " ".join((node.attrib.get("text", ""), node.attrib.get("content-desc", "")))
                    for node in root.iter("node")
                ).casefold()
                emergency_number_visible = any(
                    re.search(rf"(?<!\d){number}(?!\d)", screen_text)
                    for number in _EMERGENCY_NUMBERS
                )
                for node in root.iter("node"):
                    if node.attrib.get("clickable") != "true":
                        continue
                    if node.attrib.get("visible-to-user", "true") != "true":
                        continue
                    label = " ".join((node.attrib.get("text", ""), node.attrib.get("content-desc", ""))).strip()
                    if not label or any(word in label.casefold() for word in _EXCLUDED_LABELS):
                        continue
                    if emergency_number_visible and any(word in label.casefold() for word in ("call", "dial", "вызов", "позвон")):
                        continue
                    match = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.attrib.get("bounds", ""))
                    if not match:
                        continue
                    x1, y1, x2, y2 = map(int, match.groups())
                    w, h = x2 - x1, y2 - y1
                    if w < 24 or h < 24 or w > width * 0.92 or h > height * 0.35:
                        continue
                    point = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if point in seen:
                        continue
                    seen.add(point)
                    add({"name": "tap", "arguments": {"x": point[0], "y": point[1]}}, (x1, y1, x2, y2), 0)
                    if len(candidates) >= 36:
                        break
            except ET.ParseError:
                pass

        mid_x = width // 2
        add(
            {"name": "swipe", "arguments": {"x1": mid_x, "y1": int(height * 0.76), "x2": mid_x, "y2": int(height * 0.30), "duration_ms": 420}},
            (mid_x - 1, int(height * 0.3), mid_x + 1, int(height * 0.76)), 1,
        )
        add(
            {"name": "swipe", "arguments": {"x1": mid_x, "y1": int(height * 0.30), "x2": mid_x, "y2": int(height * 0.76), "duration_ms": 420}},
            (mid_x - 1, int(height * 0.3), mid_x + 1, int(height * 0.76)), 2,
        )
        add(
            {"name": "press_key", "arguments": {"key": "back", "long_press": False}},
            (0, 0, width, height), 3,
        )
        return candidates

    def run_task(
        self,
        task: str,
        screenshot: Callable[[], bytes | None],
        execute_tool: Callable[[str, dict[str, Any]], str],
        report: Callable[[str], None],
        max_steps: int = 8,
        autonomous: bool = False,
        curiosity: bool = False,
        conversation: list[dict[str, str]] | None = None,
        should_stop: Callable[[], bool] | None = None,
        screen_nodes: Callable[[], str] | None = None,
    ) -> str:
        del task, autonomous, conversation
        if not curiosity:
            return "Случайно инициализированная Torch-модель не понимает текстовые команды. Включите режим любопытства, чтобы исследовать экран и обучать её на изменениях интерфейса."

        self._load(report)
        torch = self._torch
        max_steps = min(max(1, int(max_steps)), 8)
        completed = 0
        reward_total = 0.0

        for step in range(max_steps):
            if should_stop and should_stop():
                break
            png = screenshot()
            if not png:
                return "Кадр экрана недоступен. Проверьте подключение ADB по USB."
            before, (width, height) = _screen_tensor(torch, png, self._device)
            try:
                hierarchy = screen_nodes() if screen_nodes else None
            except Exception as error:
                hierarchy = None
                report(f"Дерево элементов недоступно, оставлены прокрутка и кнопка Назад: {error}")
            candidates = self._screen_candidates(hierarchy, width, height)
            candidate_tensor = torch.tensor([features for _, features in candidates], dtype=torch.float32, device=self._device)
            logits = self._model(before, candidate_tensor)
            probabilities = torch.softmax(logits, dim=0)
            selected = int(torch.multinomial(probabilities, 1).item())
            log_probability = torch.log(probabilities[selected].clamp_min(1e-8))
            action, _ = candidates[selected]
            name, arguments = action["name"], action["arguments"]
            if should_stop and should_stop():
                break
            try:
                result = execute_tool(name, arguments)
            except Exception as error:
                result = f"Ошибка действия: {error}"
            report(f"Torch: шаг {step + 1}/{max_steps}, {name} — {result}")
            if should_stop and should_stop():
                break

            after_png = screenshot()
            if after_png:
                after, _ = _screen_tensor(torch, after_png, self._device)
                reward = float((before - after).abs().mean().item())
            else:
                reward = 0.0
            advantage = reward - self._baseline
            loss = -log_probability * advantage
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()
            self._baseline = 0.9 * self._baseline + 0.1 * reward
            reward_total += reward
            completed += 1
            report(f"Torch: сигнал обучения по изменению экрана = {reward:.3f}.")

        self._save()
        report(f"PyTorch: веса сохранены в {Path(self.brain_path).name}.")
        if completed == 0 and should_stop and should_stop():
            return "Исследование остановлено. Модель сохранена без новых шагов."
        average = reward_total / max(completed, 1)
        return (
            f"Исследование завершено: шагов {completed}, среднее изменение экрана {average:.3f}. "
            "Веса модели сохранены. Изменение экрана — простой сигнал обучения, а не понимание смысла интерфейса."
        )
