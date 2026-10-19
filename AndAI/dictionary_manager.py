"""Download and import vocabulary dictionaries for AndAI."""

from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, BinaryIO


MAX_DICTIONARY_BYTES = 1024 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_TERMS = 200_000
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def _clean_vocabulary(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError("В словаре ожидался объект со словами и переводами.")
    result: dict[str, list[str]] = {}
    for raw_term, raw_translations in value.items():
        term = " ".join(str(raw_term).split())[:120]
        if not term:
            continue
        if isinstance(raw_translations, str):
            translations = [raw_translations]
        elif isinstance(raw_translations, (list, tuple, set)):
            translations = list(raw_translations)
        else:
            continue
        clean = list(dict.fromkeys(
            " ".join(str(item).split())[:120]
            for item in translations
            if str(item).strip()
        ))
        if clean:
            result[term] = clean[:30]
        if len(result) >= MAX_TERMS:
            break
    if not result:
        raise ValueError("В файле не найдено ни одной пары слов и переводов.")
    return result


def _tei_vocabulary_stream(stream: BinaryIO) -> dict[str, list[str]]:
    vocabulary: dict[str, list[str]] = {}
    parents: list[ET.Element] = []
    try:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                parents.append(element)
                continue
            is_entry = element.tag.rsplit("}", 1)[-1] == "entry"
            words: list[str] = []
            translations: list[str] = []
            if is_entry:
                for child in element.iter():
                    tag = child.tag.rsplit("}", 1)[-1]
                    text = " ".join("".join(child.itertext()).split())
                    if not text:
                        continue
                    if tag == "orth":
                        words.append(text)
                    elif tag in {"quote", "def", "term"}:
                        translations.append(text)
                translations = list(dict.fromkeys(item for item in translations if item not in words))
                if words and translations:
                    for word in words:
                        existing = vocabulary.setdefault(word, [])
                        for translation in translations:
                            if translation not in existing:
                                existing.append(translation)
                element.clear()
                if len(parents) > 1:
                    parents[-2].remove(element)
            if parents:
                parents.pop()
            if is_entry and len(vocabulary) >= MAX_TERMS:
                break
    except ET.ParseError as error:
        raise ValueError(f"Не удалось прочитать XML/TEI: {error}") from None
    return _clean_vocabulary(vocabulary)


def parse_dictionary(raw: bytes, filename: str = "dictionary") -> dict[str, list[str]]:
    if len(raw) > MAX_DICTIONARY_BYTES:
        raise ValueError("Файл словаря больше 1 ГБ.")
    suffix = Path(filename.split("?", 1)[0]).suffix.casefold()
    stripped = raw.lstrip()
    if suffix in {".xml", ".tei"} or stripped.startswith(b"<"):
        return _tei_vocabulary_stream(io.BytesIO(raw))
    if suffix == ".json" or stripped.startswith((b"{", b"[")):
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("JSON-словарь больше 128 МБ; для больших словарей используйте TEI/XML, CSV или TXT.")
        try:
            data = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Не удалось прочитать JSON-словарь: {error}") from None
        if isinstance(data, dict) and "vocabulary" in data:
            data = data["vocabulary"]
        elif isinstance(data, dict) and isinstance(data.get("entries"), list):
            entries: dict[str, list[str]] = {}
            for entry in data["entries"]:
                if not isinstance(entry, dict):
                    continue
                term = entry.get("word") or entry.get("term") or entry.get("headword")
                translations = entry.get("translations") or entry.get("translation") or entry.get("meanings")
                if term and translations:
                    entries[str(term)] = translations if isinstance(translations, list) else [str(translations)]
            data = entries
        return _clean_vocabulary(data)

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        rows = csv.reader(io.StringIO(text), dialect)
        vocabulary = {
            row[0].strip(): [row[1].strip()]
            for row in rows
            if len(row) >= 2 and row[0].strip() and row[1].strip()
        }
    except csv.Error:
        vocabulary = {}
    if not vocabulary:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s*(?:\t|\s+[–—-]\s+)\s*", line, maxsplit=1)
            if len(parts) == 2 and all(parts):
                vocabulary[parts[0]] = [parts[1]]
    return _clean_vocabulary(vocabulary)


def _text_vocabulary_stream(stream: BinaryIO, prefix: bytes) -> dict[str, list[str]]:
    try:
        prefix.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        encoding = "latin-1"
    text_stream = io.TextIOWrapper(stream, encoding=encoding, errors="replace", newline="")
    sample = text_stream.read(4096)
    text_stream.seek(0)
    vocabulary: dict[str, list[str]] = {}
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        for row in csv.reader(text_stream, dialect):
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                vocabulary[row[0].strip()] = [row[1].strip()]
                if len(vocabulary) >= MAX_TERMS:
                    break
    except csv.Error:
        text_stream.seek(0)
    if not vocabulary:
        text_stream.seek(0)
        for raw_line in text_stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s*(?:\t|\s+[–—-]\s+)\s*", line, maxsplit=1)
            if len(parts) == 2 and all(parts):
                vocabulary[parts[0]] = [parts[1]]
                if len(vocabulary) >= MAX_TERMS:
                    break
    text_stream.detach()
    return _clean_vocabulary(vocabulary)


def _parse_dictionary_stream(stream: BinaryIO, filename: str) -> dict[str, list[str]]:
    prefix = stream.read(4096)
    stream.seek(0)
    suffix = Path(filename.split("?", 1)[0]).suffix.casefold()
    if suffix in {".xml", ".tei"} or prefix.lstrip().startswith(b"<"):
        return _tei_vocabulary_stream(stream)
    if suffix == ".json" or prefix.lstrip().startswith((b"{", b"[")):
        raw = stream.read(MAX_JSON_BYTES + 1)
        return parse_dictionary(raw, filename)
    return _text_vocabulary_stream(stream, prefix)


def read_dictionary_file(path: str | Path) -> dict[str, list[str]]:
    source = Path(path).expanduser()
    if source.stat().st_size > MAX_DICTIONARY_BYTES:
        raise ValueError("Файл словаря больше 1 ГБ.")
    with source.open("rb") as stream:
        return _parse_dictionary_stream(stream, source.name)


def download_dictionary(url: str) -> dict[str, list[str]]:
    url = url.strip()
    if not url.startswith(("https://", "http://")):
        raise ValueError("Введите ссылку, начинающуюся с http:// или https://.")
    request = urllib.request.Request(url, headers={"User-Agent": "AndAI dictionary importer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, tempfile.TemporaryFile(mode="w+b") as stream:
            total = 0
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DICTIONARY_BYTES:
                    raise ValueError("Файл словаря больше 1 ГБ.")
                stream.write(chunk)
            stream.seek(0)
            return _parse_dictionary_stream(stream, url)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Сервер вернул HTTP {error.code}.") from None
    except urllib.error.URLError as error:
        raise ValueError(f"Не удалось скачать словарь: {error.reason}") from None


def merge_vocabulary(path: str | Path, incoming: dict[str, list[str]]) -> int:
    target = Path(path)
    current: dict[str, Any] = {}
    if target.is_file():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("vocabulary"), dict):
                current = data["vocabulary"]
        except (OSError, json.JSONDecodeError):
            current = {}
    merged: dict[str, list[str]] = {}
    for source in (current, incoming):
        for term, translations in source.items():
            if term not in merged and len(merged) >= MAX_TERMS:
                continue
            if isinstance(translations, str):
                translations = [translations]
            if not isinstance(translations, list):
                continue
            values = merged.setdefault(str(term), [])
            for translation in translations:
                clean = " ".join(str(translation).split())[:120]
                if clean and clean not in values:
                    values.append(clean)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"version": 1, "vocabulary": merged}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(merged)
