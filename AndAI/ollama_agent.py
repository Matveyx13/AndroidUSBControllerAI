"""Vision agent for a locally hosted Ollama model with optional tool calling."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from ai_agent import AUTONOMOUS_TOOLS, CURIOUS_TOOLS, TOOLS


def _ollama_tools(autonomous: bool, curiosity: bool = False) -> list[dict[str, Any]]:
    tools = CURIOUS_TOOLS if curiosity else TOOLS + (AUTONOMOUS_TOOLS if autonomous else [])
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


def _visible_text(value: str) -> str:
    """Strip explicit reasoning tags before showing user-visible summaries."""
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<analysis>.*?</analysis>", "", value, flags=re.IGNORECASE | re.DOTALL)
    return value.strip()


class OllamaVisionAgent:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        if not self.base_url:
            raise ValueError("Укажите адрес Ollama, например http://localhost:11434")
        if not self.model:
            raise ValueError("Укажите локальную модель, например qwen3-vl:4b")

    def _request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {"model": self.model, "messages": messages, "tools": tools, "stream": False}
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                detail = body
            raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"Не удалось подключиться к Ollama: {error.reason}") from None
        if data.get("error"):
            raise RuntimeError(f"Ollama: {data['error']}")
        message = data.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama вернула ответ без message")
        return message

    @staticmethod
    def _with_image(content: str, png: bytes | None) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "user", "content": content}
        if png:
            message["images"] = [base64.b64encode(png).decode("ascii")]
        return message

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
            policy = (
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
            policy = (
                "You are a friendly conversational assistant who can also control Android. Answer ordinary questions without phone actions. "
                "The user explicitly enabled full autonomy. If they request a phone action, complete only that requested task. "
                "You may use the supplied normal ADB actions and, when needed, adb shell or Fastboot tools. "
                "Never initiate an emergency-service call, and never dial 112, 911, 999, 000, or 110. "
                "Check the screen after actions. Do not reveal private chain-of-thought; provide only concise "
                "observations, action status, and a final summary in the user's language."
            )
        else:
            policy = (
                "You are a friendly conversational assistant who can also help control Android. Answer ordinary questions without phone actions. "
                "If the user asks for a phone action, use the supplied ADB tools and screenshot, "
                "then verify the new screen. Do not type passwords or verification codes. Do not claim success "
                "when a tool reports an error. Do not reveal private chain-of-thought; provide only concise "
                "observations, action status, and a final summary in the user's language."
            )
        first_image = screenshot()
        messages: list[dict[str, Any]] = [{"role": "system", "content": policy}]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in (conversation or [])[-16:]
            if item.get("role") in ("user", "assistant") and item.get("content")
        )
        messages.append(self._with_image(
            f"Сообщение пользователя: {task}" + ("" if first_image else "\nТекущий снимок недоступен."),
            first_image,
        ))
        available_tools = _ollama_tools(autonomous, curiosity)
        allowed_names = {item["function"]["name"] for item in available_tools}
        action_count = 0

        for step in range(max_steps + 1):
            if should_stop and should_stop():
                return "Остановлено пользователем."
            message = self._request(messages, available_tools)
            content = _visible_text(str(message.get("content", "")))
            calls = message.get("tool_calls", [])
            if not calls:
                answer = content or "Готово."
                report(f"Ollama · сводка: {answer[:500]}")
                return answer
            if step == max_steps:
                return "Остановлено: достигнут лимит шагов для локальной модели."

            assistant_message = {
                "role": "assistant",
                "content": content,
                "tool_calls": calls,
            }
            messages.append(assistant_message)
            for call in calls:
                if should_stop and should_stop():
                    return "Остановлено пользователем."
                if action_count >= max_steps:
                    return "Остановлено: достигнут лимит действий для локальной модели."
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = str(function.get("name", ""))
                arguments = function.get("arguments", {})
                try:
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not name or not isinstance(arguments, dict):
                        raise ValueError("Неверный вызов инструмента")
                    if name not in allowed_names:
                        raise ValueError("Это действие недоступно в текущем режиме.")
                    result = execute_tool(name, arguments)
                    report(f"Ollama → {name}: {result}")
                except Exception as error:
                    result = f"Ошибка действия: {error}"
                    report(f"Ollama → {name or 'неизвестный инструмент'}: {result}")
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})
                action_count += 1

            if curiosity and action_count >= max_steps:
                return "Действие выполнено; исследую следующий кадр."
            next_image = screenshot()
            if next_image:
                messages.append(self._with_image("Current phone screen after the previous action.", next_image))
            else:
                messages.append({"role": "user", "content": "A new screenshot is unavailable; the phone may have left Android."})

        return "Действия завершены."
