"""Small vision agent using the OpenAI Responses API and bounded ADB tools."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Callable


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function", "name": "tap", "description": "Tap a point on the phone screen. Coordinates use the original screenshot pixels, origin at top left.",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "swipe", "description": "Swipe from one screen point to another.",
        "parameters": {"type": "object", "properties": {"x1": {"type": "integer"}, "y1": {"type": "integer"}, "x2": {"type": "integer"}, "y2": {"type": "integer"}, "duration_ms": {"type": "integer"}}, "required": ["x1", "y1", "x2", "y2", "duration_ms"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "hold_screen", "description": "Hold a finger at one point on the screen for a duration.",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "duration_ms": {"type": "integer"}}, "required": ["x", "y", "duration_ms"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "press_key", "description": "Press a device button or navigation key; optional long press is a simulated Android key event.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string", "enum": ["power", "volume_up", "volume_down", "home", "back", "recents", "enter", "menu", "wakeup", "camera"]}, "long_press": {"type": "boolean"}}, "required": ["key", "long_press"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "type_text", "description": "Type text into the currently focused Android field.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "open_app", "description": "Open an installed Android app by package name.",
        "parameters": {"type": "object", "properties": {"package_name": {"type": "string"}}, "required": ["package_name"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "set_network", "description": "Enable or disable Wi-Fi or mobile data using Android's svc command.",
        "parameters": {"type": "object", "properties": {"feature": {"type": "string", "enum": ["wifi", "mobile_data"]}, "enabled": {"type": "boolean"}}, "required": ["feature", "enabled"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "set_bluetooth", "description": "Enable or disable Bluetooth on the Android phone.",
        "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "reboot", "description": "Reboot Android, Recovery, or the bootloader/Fastboot screen.",
        "parameters": {"type": "object", "properties": {"target": {"type": "string", "enum": ["system", "recovery", "fastboot"]}}, "required": ["target"], "additionalProperties": False}, "strict": True,
    },
]

AUTONOMOUS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function", "name": "run_adb_shell", "description": "Run an arbitrary Android adb shell command. Available in full autonomy and research mode.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "run_fastboot", "description": "Run an arbitrary Fastboot command on the selected bootloader device. Available in full autonomy and research mode.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}, "strict": True,
    },
]

CURIOUS_TOOLS: list[dict[str, Any]] = TOOLS + AUTONOMOUS_TOOLS


class OpenAIVisionAgent:
    def __init__(self, api_key: str, model: str = "gpt-6-astra") -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        if not self.api_key:
            raise ValueError("Введите API key, чтобы включить ИИ")
        if not self.model:
            raise ValueError("Укажите название vision-модели")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("error", {}).get("message", body)
            except json.JSONDecodeError:
                detail = body
            raise RuntimeError(f"OpenAI API ({error.code}): {detail}") from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"Не удалось подключиться к OpenAI API: {error.reason}") from None

    @staticmethod
    def _image_content(png: bytes) -> dict[str, str]:
        encoded = base64.b64encode(png).decode("ascii")
        return {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "high"}

    @staticmethod
    def _response_text(output: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in output:
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") in ("output_text", "text") and content.get("text"):
                        parts.append(content["text"])
        return "\n".join(parts).strip() or "Готово."

    def run_task(
        self,
        task: str,
        screenshot: Callable[[], bytes | None],
        execute_tool: Callable[[str, dict[str, Any]], str],
        report: Callable[[str], None],
        max_steps: int = 12,
        autonomous: bool = False,
        curiosity: bool = False,
        conversation: list[dict[str, str]] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        if curiosity:
            max_steps = min(max_steps, 8)
        first_image = screenshot()
        history: list[dict[str, Any]] = []
        for item in (conversation or [])[-16:]:
            role = item.get("role")
            text = item.get("content", "")
            if role in ("user", "assistant") and text:
                history.append({"role": role, "content": [{"type": "input_text", "text": text}]})
        content: list[dict[str, Any]] = [{"type": "input_text", "text": f"Сообщение пользователя: {task}"}]
        if first_image:
            content.append(self._image_content(first_image))
        else:
            content.append({"type": "input_text", "text": "Снимок экрана сейчас недоступен (возможно, телефон в Fastboot/Recovery)."})
        history.append({"role": "user", "content": content})
        if curiosity:
            instructions = (
                "Ты управляешь Android-устройством по прямому разрешению владельца. В режиме исследования разрешено выполнять любые действия "
                "через интерфейс, вводить текст, менять настройки и сеть, открывать приложения, устанавливать и удалять приложения, "
                "перезагружать устройство, использовать произвольные ADB shell и Fastboot команды, включая очистку и прошивку. "
                "Единственное исключение: никогда не инициируй, не набирай и не подтверждай вызовы экстренным службам; "
                "не используй номера 112, 911, 999, 000, 110 и не нажимай кнопки экстренного вызова. Если действие может связаться "
                "с экстренной службой, откажись от него. Для остальных действий не запрашивай подтверждение. "
                "Работай по одному шагу, проверяй новый кадр и сообщай пользователю короткие наблюдения, выполненные действия и результат. "
                "Не показывай скрытые рассуждения. Отвечай по-русски или на языке пользователя."
            )
        elif autonomous:
            instructions = (
                "Ты дружелюбный собеседник, который также умеет управлять Android. На обычные вопросы отвечай в диалоге без действий с телефоном. "
                "Пользователь явно включил полную автономию. Если он попросил действие, выполняй его без дополнительных подтверждений. "
                "Тебе доступны обычные действия, произвольные adb shell команды и Fastboot команды; при необходимости "
                "можно менять настройки, устанавливать или удалять приложения, отправлять сообщения, выполнять reboot, "
                "flash и erase. Никогда не инициируй вызов экстренным службам и не набирай номера 112, 911, 999, 000 или 110. "
                "Не выполняй действий вне поставленной пользователем задачи. Проверяй результат по экрану "
                "или выводу команды и сообщай об ошибках. Отвечай на языке пользователя (Russian/English)."
            )
        else:
            instructions = (
                "Ты дружелюбный собеседник и помощник по Android. На обычные вопросы отвечай в диалоге, не нажимая кнопки телефона. "
                "Если пользователь просит действие, управляй телефоном через ограниченный набор ADB функций. Анализируй изображение экрана, "
                "используй координаты исходного кадра и после каждого действия проверь новый кадр. Не утверждай, "
                "что действие выполнено, если инструмент вернул ошибку. Не вводи пароли, коды подтверждения и платёжные данные. "
                "Не подтверждай удаление данных, сброс телефона, установку/удаление приложений, отправку сообщений, "
                "звонки, покупки или выдачу разрешений, если пользователь прямо не попросил это. Fastboot не передаёт "
                "изображение экрана через универсальный ADB; если снимка нет, объясни это и не угадывай координаты. "
                "Понимай обычные команды на русском и английском и отвечай на языке пользователя."
            )
        available_tools = CURIOUS_TOOLS if curiosity else TOOLS + (AUTONOMOUS_TOOLS if autonomous else [])
        allowed_names = {item["name"] for item in available_tools}
        action_count = 0
        for step in range(max_steps + 1):
            if should_stop and should_stop():
                return "Остановлено пользователем."
            response = self._request({
                "model": self.model,
                "instructions": instructions,
                "input": history,
                "tools": available_tools,
                "store": False,
            })
            if response.get("error"):
                raise RuntimeError(response["error"].get("message", "API вернул ошибку"))
            output = response.get("output", [])
            calls = [item for item in output if item.get("type") == "function_call"]
            if not calls:
                return self._response_text(output)
            if step == max_steps:
                return "Остановлено: достигнут лимит действий за один запрос. " + self._response_text(output)

            # Responses API requires returning the model's output items and a result for each call.
            history.extend(output)
            for call in calls:
                if should_stop and should_stop():
                    return "Остановлено пользователем."
                if action_count >= max_steps:
                    return "Остановлено: достигнут лимит действий за один запрос."
                name = call.get("name", "")
                try:
                    if name not in allowed_names:
                        raise ValueError("Это действие недоступно в текущем режиме.")
                    arguments = json.loads(call.get("arguments", "{}"))
                    if not isinstance(arguments, dict):
                        raise ValueError("Ожидались параметры в формате object")
                    result = execute_tool(name, arguments)
                    report(f"AI → {name}: {result}")
                except Exception as error:  # Return tool errors so the model can recover.
                    result = f"Ошибка действия: {error}"
                    report(f"AI → {name}: {result}")
                history.append({"type": "function_call_output", "call_id": call["call_id"], "output": result})
                action_count += 1

            if curiosity and action_count >= max_steps:
                return "Действие выполнено; исследую следующий кадр."
            next_image = screenshot()
            if next_image:
                history.append({"role": "user", "content": [
                    {"type": "input_text", "text": "Текущий экран после предыдущего действия. Продолжай задание по этому состоянию."},
                    self._image_content(next_image),
                ]})
            else:
                history.append({"role": "user", "content": "Новый кадр недоступен. Телефон мог перейти в bootloader/Fastboot."})
        return "Действия завершены."

