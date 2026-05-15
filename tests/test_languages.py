import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kugelaudio_open.languages import (
    build_language_instruction,
    get_language,
    get_language_dropdown_choices,
    get_supported_language_codes,
    parse_language_dropdown_choice,
)


class LanguageTests(unittest.TestCase):
    def test_supported_language_codes_include_expected_entries(self):
        codes = get_supported_language_codes()
        self.assertIn("en", codes)
        self.assertIn("de", codes)
        self.assertIn("sr", codes)

    def test_get_language_returns_metadata(self):
        language = get_language("de")
        self.assertIsNotNone(language)
        self.assertEqual(language.name, "German")
        self.assertEqual(language.flag, "🇩🇪")

    def test_build_language_instruction_is_empty_for_auto(self):
        self.assertEqual(build_language_instruction(None), "")
        self.assertEqual(build_language_instruction("auto"), "")

    def test_build_language_instruction_formats_selected_language(self):
        self.assertEqual(build_language_instruction("sr"), " Output the speech in Serbian (sr).")

    def test_build_language_instruction_rejects_unknown_language(self):
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            build_language_instruction("zz")

    def test_dropdown_choices_include_auto_detect(self):
        choices = get_language_dropdown_choices()
        self.assertEqual(choices[0], "Auto-detect")
        self.assertIn("🇩🇪 German (de)", choices)

    def test_parse_dropdown_choice(self):
        self.assertEqual(parse_language_dropdown_choice("🇩🇪 German (de)"), "de")
        self.assertIsNone(parse_language_dropdown_choice("Auto-detect"))
        self.assertIsNone(parse_language_dropdown_choice("garbage"))


if __name__ == "__main__":
    unittest.main()
