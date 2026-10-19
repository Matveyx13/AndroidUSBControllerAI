"""A small local, user-trainable command brain stored as JSON rules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable


DICTIONARY_PATH = Path(__file__).resolve().with_name("dictionary.json")
DOWNLOADED_DICTIONARY_PATH = Path(__file__).resolve().with_name("downloaded_dictionary.json")
DICTIONARY: dict[str, Any] = {}
VOCABULARY: dict[str, Any] = {}
VOCABULARY_INDEX: dict[tuple[str, ...], tuple[str, str]] = {}
VOCABULARY_MAX_TOKENS = 1


def _vocabulary_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+|[^\w\s]+", text.casefold(), flags=re.UNICODE))


def reload_dictionary() -> int:
    """Reload bundled and imported words while keeping imported dict references valid."""
    global DICTIONARY_DATA, VOCABULARY_MAX_TOKENS, VOCABULARY_INDEX
    try:
        DICTIONARY_DATA = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        commands = DICTIONARY_DATA.get("commands", {})
        vocabulary = dict(DICTIONARY_DATA.get("vocabulary", {}))
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        DICTIONARY_DATA = {}
        commands = {}
        vocabulary = {}
    try:
        imported = json.loads(DOWNLOADED_DICTIONARY_PATH.read_text(encoding="utf-8"))
        imported = imported.get("vocabulary", {}) if isinstance(imported, dict) else {}
        for term, translations in imported.items():
            old = vocabulary.get(term, [])
            if isinstance(old, str):
                old = [old]
            if isinstance(translations, str):
                translations = [translations]
            vocabulary[term] = list(dict.fromkeys([*old, *(translations if isinstance(translations, list) else [])]))
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        pass
    new_index: dict[tuple[str, ...], tuple[str, str]] = {}
    new_max_tokens = 1
    for russian, translations in vocabulary.items():
        terms = [russian, *(translations if isinstance(translations, (list, tuple)) else [translations])]
        for term in terms:
            tokens = _vocabulary_tokens(str(term))
            if not tokens:
                continue
            new_index.setdefault(tokens, (str(russian), str(term)))
            new_max_tokens = max(new_max_tokens, len(tokens))
    DICTIONARY.clear()
    DICTIONARY.update(commands if isinstance(commands, dict) else {})
    VOCABULARY.clear()
    VOCABULARY.update(vocabulary)
    VOCABULARY_INDEX = new_index
    VOCABULARY_MAX_TOKENS = new_max_tokens
    return len(VOCABULARY)


reload_dictionary()


def _has_command(text: str, key: str) -> bool:
    return any(phrase.casefold() in text.casefold() for phrase in DICTIONARY.get(key, []))


def _alternation(key: str) -> str:
    phrases = sorted(DICTIONARY.get(key, []), key=len, reverse=True)
    escaped = [re.escape(phrase).replace(r"\ ", r"\s+") for phrase in phrases]
    return "(?:" + "|".join(escaped) + ")" if escaped else r"(?!)"


def _normalize_vocabulary(text: str) -> str:
    tokens = list(_vocabulary_tokens(text))
    normalized_parts: list[str] = []
    index = 0
    while index < len(tokens):
        match: tuple[str, str] | None = None
        matched_end = index + 1
        for end in range(min(len(tokens), index + VOCABULARY_MAX_TOKENS), index, -1):
            match = VOCABULARY_INDEX.get(tuple(tokens[index:end]))
            if match is not None:
                matched_end = end
                break
        if match is not None:
            normalized_parts.append(match[0].casefold())
            index = matched_end
        else:
            normalized_parts.append(tokens[index])
            index += 1
    normalized = " ".join(normalized_parts)
    # Normalize whole command phrases too, so a rule taught in one language can
    # also match its Russian/English dictionary equivalent.
    for command, phrases in DICTIONARY.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            pattern = rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)"
            normalized = re.sub(pattern, f" {command} ", normalized)
    return " ".join(normalized.split())


def dictionary_matches(text: str) -> list[str]:
    """Return bilingual dictionary entries that occur as whole terms."""
    tokens = _vocabulary_tokens(text)
    matches: list[str] = []
    found_heads: set[str] = set()
    for start in range(len(tokens)):
        for end in range(min(len(tokens), start + VOCABULARY_MAX_TOKENS), start, -1):
            match = VOCABULARY_INDEX.get(tuple(tokens[start:end]))
            if match and match[0] not in found_heads:
                matches.append(f"{match[0]} ({match[1]})")
                found_heads.add(match[0])
    for command, phrases in DICTIONARY.items():
        found = next((phrase for phrase in phrases if re.search(
            rf"(?<!\w){re.escape(str(phrase).casefold())}(?!\w)", text.casefold()
        )), None)
        if found is not None:
            matches.append(f"команда {command} ({found})")
    return matches


KEY_ALIASES = {
    "питание": "power", "power": "power", "вкл": "power",
    "громкость+": "volume_up", "громкость_вверх": "volume_up", "volume_up": "volume_up", "volumeup": "volume_up",
    "громкость-": "volume_down", "громкость_вниз": "volume_down", "volume_down": "volume_down", "volumedown": "volume_down",
    "домой": "home", "home": "home", "назад": "back", "back": "back",
    "недавние": "recents", "recents": "recents", "enter": "enter", "ввод": "enter",
}


def parse_script(script: str, *, allow_autonomous: bool = False) -> list[dict[str, Any]]:
    """Parse a safe, one-action-per-line teaching script."""
    actions: list[dict[str, Any]] = []
    for line_number, raw in enumerate(script.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pieces = line.split(maxsplit=1)
        command = pieces[0].lower()
        rest = pieces[1].strip() if len(pieces) > 1 else ""
        values = rest.split()
        try:
            if command == "shell" and rest:
                if not allow_autonomous:
                    raise ValueError("команды shell доступны только при включённой полной автономии")
                action = {"name": "run_adb_shell", "arguments": {"command": rest}}
            elif command == "adb" and re.match(r"^shell\s+.+", rest, re.IGNORECASE):
                if not allow_autonomous:
                    raise ValueError("команды shell доступны только при включённой полной автономии")
                shell_command = re.sub(r"^shell\s+", "", rest, count=1, flags=re.IGNORECASE)
                action = {"name": "run_adb_shell", "arguments": {"command": shell_command}}
            elif command == "fastboot" and rest:
                if not allow_autonomous:
                    raise ValueError("команды Fastboot доступны только при включённой полной автономии")
                action = {"name": "run_fastboot", "arguments": {"command": rest}}
            elif command == "tap" and len(values) == 2:
                action = {"name": "tap", "arguments": {"x": int(values[0]), "y": int(values[1])}}
            elif command == "swipe" and len(values) in (4, 5):
                nums = [int(value) for value in values]
                action = {"name": "swipe", "arguments": {"x1": nums[0], "y1": nums[1], "x2": nums[2], "y2": nums[3], "duration_ms": nums[4] if len(nums) == 5 else 400}}
            elif command in ("hold", "удерживать") and len(values) in (2, 3):
                nums = [int(value) for value in values]
                action = {"name": "hold_screen", "arguments": {"x": nums[0], "y": nums[1], "duration_ms": nums[2] if len(nums) == 3 else 1000}}
            elif command in ("press", "кнопка") and len(values) in (1, 2):
                key = KEY_ALIASES.get(values[0].lower())
                if not key:
                    raise ValueError("неизвестная кнопка")
                long_press = len(values) == 2 and values[1].lower() in ("long", "hold", "долго", "зажать")
                action = {"name": "press_key", "arguments": {"key": key, "long_press": long_press}}
            elif command in ("wifi", "data", "bluetooth", "bt") and len(values) == 1 and values[0].lower() in ("on", "off", "вкл", "выкл", "включить", "выключить"):
                enabled = values[0].lower() in ("on", "вкл", "включить")
                if command in ("bluetooth", "bt"):
                    action = {"name": "set_bluetooth", "arguments": {"enabled": enabled}}
                else:
                    action = {"name": "set_network", "arguments": {"feature": "wifi" if command == "wifi" else "mobile_data", "enabled": enabled}}
            elif command in ("reboot", "перезагрузка") and len(values) == 1:
                target = values[0].lower()
                target = {"android": "system", "система": "system", "загрузчик": "fastboot", "bootloader": "fastboot", "fastboot": "fastboot", "рекавери": "recovery"}.get(target, target)
                if target not in ("system", "fastboot", "recovery"):
                    raise ValueError("укажите system, fastboot или recovery")
                action = {"name": "reboot", "arguments": {"target": target}}
            elif command in ("type", "text", "текст") and rest:
                action = {"name": "type_text", "arguments": {"text": rest}}
            elif command in ("app", "приложение") and len(values) == 1:
                action = {"name": "open_app", "arguments": {"package_name": values[0]}}
            else:
                raise ValueError("неверный формат команды")
        except ValueError as error:
            raise ValueError(f"строка {line_number}: {error}") from None
        actions.append(action)
    if not actions:
        raise ValueError("Добавьте хотя бы одно действие")
    return actions


class LocalBrain:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.rules: list[dict[str, Any]] = []
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                rules = data.get("rules", [])
                if not isinstance(rules, list):
                    raise ValueError("Поле rules должно быть списком")
                if any(
                    not isinstance(rule, dict)
                    or not isinstance(rule.get("phrase"), str)
                    or not isinstance(rule.get("actions"), list)
                    for rule in rules
                ):
                    raise ValueError("В мозге ожидались правила с phrase и actions")
                self.rules = rules
            except (OSError, json.JSONDecodeError, AttributeError) as error:
                raise ValueError(f"Не удалось прочитать файл мозга: {error}") from None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "rules": self.rules}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_rule(self, phrase: str, script: str, *, allow_autonomous: bool = False) -> int:
        phrase = " ".join(phrase.casefold().split())
        if len(phrase) < 2:
            raise ValueError("Введите фразу длиной хотя бы два символа")
        actions = parse_script(script, allow_autonomous=allow_autonomous)
        self.rules = [rule for rule in self.rules if str(rule.get("phrase", "")).casefold() != phrase]
        self.rules.append({"phrase": phrase, "actions": actions})
        self.save()
        return len(self.rules)

    def add_actions(self, phrase: str, actions: list[dict[str, Any]]) -> int:
        """Store a demonstration using only bounded, repeatable phone actions."""
        phrase = " ".join(phrase.casefold().split())
        if len(phrase) < 2:
            raise ValueError("Введите фразу длиной хотя бы два символа")
        if not isinstance(actions, list) or not 1 <= len(actions) <= 100:
            raise ValueError("В демонстрации должно быть от 1 до 100 действий")

        safe_actions: list[dict[str, Any]] = []
        allowed_keys = {"power", "volume_up", "volume_down", "home", "back", "recents"}
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("Некорректная запись действия")
            name = action.get("name")
            args = action.get("arguments")
            if not isinstance(args, dict):
                raise ValueError("У действия нет параметров")
            try:
                if name == "tap":
                    clean = {"x": int(args["x"]), "y": int(args["y"])}
                    if any(not 0 <= value <= 20000 for value in clean.values()):
                        raise ValueError
                elif name == "swipe":
                    clean = {key: int(args[key]) for key in ("x1", "y1", "x2", "y2")}
                    clean["duration_ms"] = min(10000, max(80, int(args.get("duration_ms", 400))))
                    if any(not 0 <= clean[key] <= 20000 for key in ("x1", "y1", "x2", "y2")):
                        raise ValueError
                elif name == "hold_screen":
                    clean = {"x": int(args["x"]), "y": int(args["y"])}
                    clean["duration_ms"] = min(10000, max(100, int(args.get("duration_ms", 1000))))
                    if any(not 0 <= clean[key] <= 20000 for key in ("x", "y")):
                        raise ValueError
                elif name == "press_key" and args.get("key") in allowed_keys:
                    clean = {"key": args["key"], "long_press": bool(args.get("long_press", False))}
                elif name == "open_app" and re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", str(args.get("package_name", ""))):
                    clean = {"package_name": str(args["package_name"])}
                elif name == "set_network" and args.get("feature") in ("wifi", "mobile_data") and isinstance(args.get("enabled"), bool):
                    clean = {"feature": args["feature"], "enabled": args["enabled"]}
                elif name == "set_bluetooth" and isinstance(args.get("enabled"), bool):
                    clean = {"enabled": args["enabled"]}
                else:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Нельзя сохранить небезопасное или неверное действие: {name}") from None
            safe_actions.append({"name": name, "arguments": clean})

        self.rules = [rule for rule in self.rules if str(rule.get("phrase", "")).casefold() != phrase]
        self.rules.append({"phrase": phrase, "actions": safe_actions})
        self.save()
        return len(self.rules)

    def _find_rule(self, task: str) -> dict[str, Any] | None:
        normalized = _normalize_vocabulary(task)
        matches = [
            rule for rule in self.rules
            if _normalize_vocabulary(str(rule.get("phrase", ""))) in normalized
        ]
        return max(matches, key=lambda rule: len(str(rule.get("phrase", ""))), default=None)

    @staticmethod
    def _builtin_actions(task: str) -> list[dict[str, Any]] | None:
        lowered = task.casefold()
        if _has_command(lowered, "bluetooth_on"):
            return [{"name": "set_bluetooth", "arguments": {"enabled": True}}]
        if _has_command(lowered, "bluetooth_off"):
            return [{"name": "set_bluetooth", "arguments": {"enabled": False}}]
        if _has_command(lowered, "wifi_on"):
            return [{"name": "set_network", "arguments": {"feature": "wifi", "enabled": True}}]
        if _has_command(lowered, "wifi_off"):
            return [{"name": "set_network", "arguments": {"feature": "wifi", "enabled": False}}]
        if _has_command(lowered, "data_on"):
            return [{"name": "set_network", "arguments": {"feature": "mobile_data", "enabled": True}}]
        if _has_command(lowered, "data_off"):
            return [{"name": "set_network", "arguments": {"feature": "mobile_data", "enabled": False}}]
        if _has_command(lowered, "bootloader"):
            return [{"name": "reboot", "arguments": {"target": "fastboot"}}]
        if _has_command(lowered, "recovery"):
            return [{"name": "reboot", "arguments": {"target": "recovery"}}]
        if _has_command(lowered, "reboot"):
            return [{"name": "reboot", "arguments": {"target": "system"}}]

        key_names = ("power", "volume_up", "volume_down", "home", "back", "recents")
        if _has_command(lowered, "press") or _has_command(lowered, "long_press") or any(_has_command(lowered, key) for key in ("home", "back", "recents", "volume_up", "volume_down")):
            for key in key_names:
                if _has_command(lowered, key):
                    long_press = _has_command(lowered, "long_press")
                    return [{"name": "press_key", "arguments": {"key": key, "long_press": long_press}}]

        tap = re.search(rf"{_alternation('tap')}\s*[:,]?\s*(\d{{1,5}})[ ,]+(\d{{1,5}})", lowered)
        if tap:
            return [{"name": "tap", "arguments": {"x": int(tap.group(1)), "y": int(tap.group(2))}}]
        swipe = re.search(rf"{_alternation('swipe')}\s+(\d{{1,5}})[ ,]+(\d{{1,5}})[ ,]+(\d{{1,5}})[ ,]+(\d{{1,5}})(?:[ ,]+(\d{{1,5}}))?", lowered)
        if swipe:
            values = [int(value) if value else None for value in swipe.groups()]
            return [{"name": "swipe", "arguments": {"x1": values[0], "y1": values[1], "x2": values[2], "y2": values[3], "duration_ms": values[4] or 400}}]
        hold = re.search(rf"{_alternation('long_press')}\s+(?:screen\s+|экран\s+)?(\d{{1,5}})[ ,]+(\d{{1,5}})(?:[ ,]+(\d{{1,5}}))?", lowered)
        if hold:
            return [{"name": "hold_screen", "arguments": {"x": int(hold.group(1)), "y": int(hold.group(2)), "duration_ms": int(hold.group(3) or 1000)}}]
        text = re.search(rf"{_alternation('type')}\s+(.+)", task, re.IGNORECASE)
        if text:
            return [{"name": "type_text", "arguments": {"text": text.group(1)}}]
        if re.search(r"\b(open|launch)\s+settings\b|открой\s+настройки", lowered):
            return [{"name": "open_app", "arguments": {"package_name": "com.android.settings"}}]
        return None

    def run_task(
        self,
        task: str,
        execute_tool: Callable[[str, dict[str, Any]], str],
        report: Callable[[str], None],
        *,
        allow_autonomous: bool = False,
    ) -> str:
        rule = self._find_rule(task)
        direct_command = bool(re.match(r"^(?:shell\s+|adb\s+shell\s+|fastboot\s+)", task.strip(), re.IGNORECASE))
        if rule:
            actions = rule.get("actions", [])
            source = f"обученное правило «{rule['phrase']}»"
        elif direct_command:
            if not allow_autonomous:
                return "Эта команда требует включить «Полную автономию» для доступа к ADB shell/Fastboot."
            try:
                actions = parse_script(task.strip(), allow_autonomous=True)
                source = "прямая автономная команда"
            except ValueError as error:
                return f"Не удалось разобрать команду: {error}"
        else:
            actions = self._builtin_actions(task)
            source = "простая команда"
        privileged_actions = {"run_adb_shell", "run_fastboot"}
        if not allow_autonomous and any(action.get("name") in privileged_actions for action in actions or []):
            return "Это правило требует включить «Полную автономию» для доступа к ADB shell/Fastboot."
        if not actions:
            return (
                "Локальный мозг на правилах пока не знает такую фразу и не умеет вести свободный диалог или анализировать экран. "
                "Для разговора и режима любопытства выберите Ollama / Qwen; для этой команды можно нажать «Обучить пример»."
            )
        completed: list[str] = []
        for action in actions:
            try:
                result = execute_tool(action["name"], action["arguments"])
                completed.append(result)
                report(f"Локальный мозг → {action['name']}: {result}")
            except Exception as error:
                report(f"Локальный мозг: ошибка — {error}")
                return f"Остановлено на ошибке: {error}"
        return f"Выполнено ({source}): " + "; ".join(completed)

