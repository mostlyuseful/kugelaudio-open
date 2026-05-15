"""Language metadata and helpers for KugelAudio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

LanguageQuality = Literal["high", "medium", "limited"]


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    flag: str
    quality: LanguageQuality

    def display_label(self) -> str:
        quality_suffix = {
            "high": "",
            "medium": " · medium",
            "limited": " · limited ⚠️",
        }[self.quality]
        return f"{self.flag} {self.name} ({self.code}){quality_suffix}"


_LANGUAGES: List[Language] = [
    Language("en", "English", "🇺🇸", "high"),
    Language("de", "German", "🇩🇪", "high"),
    Language("fr", "French", "🇫🇷", "high"),
    Language("es", "Spanish", "🇪🇸", "high"),
    Language("it", "Italian", "🇮🇹", "medium"),
    Language("pt", "Portuguese", "🇵🇹", "medium"),
    Language("nl", "Dutch", "🇳🇱", "medium"),
    Language("pl", "Polish", "🇵🇱", "medium"),
    Language("ru", "Russian", "🇷🇺", "medium"),
    Language("uk", "Ukrainian", "🇺🇦", "limited"),
    Language("cs", "Czech", "🇨🇿", "limited"),
    Language("ro", "Romanian", "🇷🇴", "limited"),
    Language("hu", "Hungarian", "🇭🇺", "limited"),
    Language("sv", "Swedish", "🇸🇪", "limited"),
    Language("da", "Danish", "🇩🇰", "limited"),
    Language("fi", "Finnish", "🇫🇮", "limited"),
    Language("no", "Norwegian", "🇳🇴", "limited"),
    Language("el", "Greek", "🇬🇷", "limited"),
    Language("bg", "Bulgarian", "🇧🇬", "limited"),
    Language("sk", "Slovak", "🇸🇰", "limited"),
    Language("hr", "Croatian", "🇭🇷", "limited"),
    Language("sr", "Serbian", "🇷🇸", "limited"),
    Language("tr", "Turkish", "🇹🇷", "limited"),
]

_DEFAULT_LANGUAGE = "auto"
_BY_CODE: Dict[str, Language] = {language.code: language for language in _LANGUAGES}


def get_supported_languages() -> List[Language]:
    return list(_LANGUAGES)


def get_language(code: str) -> Optional[Language]:
    return _BY_CODE.get(code.lower())


def get_supported_language_codes() -> List[str]:
    return [language.code for language in _LANGUAGES]


def normalize_language_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = language.strip().lower()
    if not normalized or normalized == _DEFAULT_LANGUAGE:
        return None
    if normalized not in _BY_CODE:
        raise ValueError(
            f"Unsupported language '{language}'. Supported languages: {', '.join(get_supported_language_codes())}"
        )
    return normalized


def build_language_instruction(language: Optional[str]) -> str:
    normalized = normalize_language_code(language)
    if normalized is None:
        return ""
    selected = _BY_CODE[normalized]
    return f" Output the speech in {selected.name} ({selected.code})."


def get_language_dropdown_choices() -> List[str]:
    return ["Auto-detect"] + [language.display_label() for language in _LANGUAGES]


def parse_language_dropdown_choice(choice: Optional[str]) -> Optional[str]:
    if choice is None:
        return None
    normalized = choice.strip()
    if not normalized or normalized == "Auto-detect":
        return None
    code = normalized.split("(")[-1].split(")")[0].strip().lower()
    return code if code in _BY_CODE else None
