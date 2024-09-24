# Copyright 2024 The OpenAI team and The HuggingFace Team. All rights reserved.
# Most of the code is copy pasted from the original whisper repository
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union


# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]

        elif unicodedata.category(char) == "Mn":
            return ""

        elif unicodedata.category(char)[0] in "MSP":
            return " "

        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))


class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        # remove words between brackets
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(re.findall(r"\X", s, re.U))

        # replace any successive whitespace characters with a space
        s = re.sub(r"\s+", " ", s)

        return s


class DutchNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, while handling:

    - remove any commas
    - keep the suffixes such as: `1960s`, `274e`, etc.
    - spell out currency symbols after the number. e.g. `€20 miljoen` -> `20000000 euro`
    - interpret successive single-digit numbers as nominal: `een nul een` -> `101`
    """

    def __init__(self):
        super().__init__()

        self.zeros = {"nul"}
        # fmt: off
        self.ones = {
            name: i
            for i, name in enumerate(
                ["één", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen", "tien", "elf", "twaalf", "dertien", "veertien", "vijftien", "zestien", "zeventien", "achttien", "negentien"],
                start=1,
            )
        }
        # fmt: on
        self.ones_ordinal = {
            "eerste": (1, "e"),
            "tweede": (2, "e"),
            "derde": (3, "e"),
            "vierde": (4, "e"),
            "vijfde": (5, "e"),
            "zesde": (6, "e"),
            "zevende": (7, "e"),
            "achtste": (8, "e"),
            "negende": (9, "e"),
            "tiende": (10, "e"),
            "elfde": (11, "e"),
            "twaalfde": (12, "e"),
            **{
                name + "de": (value, "de")
                for name, value in self.ones.items()
                if value > 12
            },
        }
        self.ones_suffixed = {**self.ones_ordinal}

        self.tens = {
            "twintig": 20,
            "dertig": 30,
            "veertig": 40,
            "vijftig": 50,
            "zestig": 60,
            "zeventig": 70,
            "tachtig": 80,
            "negentig": 90,
        }
        self.tens_ordinal = {
            name + "ste": (value, "ste") for name, value in self.tens.items()}
        self.tens_suffixed = {**self.tens_ordinal}

        self.multipliers = {
            "honderd": 100,
            "duizend": 1_000,
            "miljoen": 1_000_000,
            "miljard": 1_000_000_000,
            "biljoen": 1_000_000_000_000,
            "biljard": 1_000_000_000_000_000,
            "triljoen": 1_000_000_000_000_000_000,
        }
        self.multipliers_ordinal = {
            name + "ste": (value, "ste") for name, value in self.multipliers.items()}
        self.multipliers_suffixed = {**self.multipliers_ordinal}
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "min": "-",
            "negatief": "-",
            "plus": "+",
            "positief": "+",
        }
        self.following_prefixers = {
            "pond": "£",
            "ponds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes = set(list(self.preceding_prefixers.values(
        )) + list(self.following_prefixers.values()))
        self.suffixers = {
            "pro": {"cent": "%"},
            "procent": "%",
        }
        self.specials = {"en", "dubbel", "punt"}

        self.words = {
            key
            for mapping in [
                self.zeros,
                self.ones,
                self.ones_suffixed,
                self.tens,
                self.tens_suffixed,
                self.multipliers,
                self.multipliers_suffixed,
                self.preceding_prefixers,
                self.following_prefixers,
                self.suffixers,
                self.specials,
            ]
            for key in mapping
        }

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for i, current in enumerate(words):
            prev = words[i - 1] if i != 0 else None
            next = words[i + 1] if i != len(words) - 1 else None
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(
                r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f is None:
                    raise ValueError("Converting the fraction failed")

                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:  # replace the last zero with the digit
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # elf to negentien
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                # ordinal or cardinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # elf to negentien
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                # ordinal or cardinal; yield the number right away
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                # apply prefix (positive, min, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    # apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "en":
                    # ignore "en" after hundreds, thousands, etc.
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "dubbel" or current == "trippel":
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "dubbel" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "punt":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # replace "<number> en een half" with "<number> punt vijf"
        results = []

        segments = re.split(r"\ben\s+een\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("punt vijf")
                else:
                    results.append("en een half")

        s = " ".join(results)

        # put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # but remove spaces which could be a suffix
        s = re.sub(r"([0-9])\s+(e|ste|de)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # apply currency postprocessing; "$2 en ¢7" -> "$2.07"
        s = re.sub(
            r"([€£$])([0-9]+) (?:en )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(
            s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class DutchSpellingNormalizer:
    """
    Placeholder for spelling normalization. In Dutch, there is less emphasis on dialect normalization 
    compared to English, but this class can be extended to include specific mappings if needed.
    """

    def __init__(self, dutch_spelling_mapping):
        self.mapping = dutch_spelling_mapping

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class DutchTextNormalizer:
    def __init__(self, dutch_spelling_mapping={}):
        self.ignore_patterns = r"\b(uh|um|mmm|hmm)\b"
        self.replacers = {
            # common contractions and phrases
            r"\bd'r\b": "haar",
            r"\b't\b": "het",
            r"\bm'n\b": "mijn",
            r"\bjij bent\b": "je bent",
            r"\bik ben\b": "ik ben",
            r"\bdhr\b": "meneer ",
            r"\bmevr\b": "mevrouw ",
            r"\bst\b": "sint ",
            r"\bdr\b": "doctor ",
            r"\bdrs\b": "doctorandus ",
            r"\bing\b": "ingenieur ",
            r"\bdrs\b": "doctorandus ",
            r"\bprof\b": "professor ",
            r"\b&\b": " en ",
        }
        self.standardize_numbers = DutchNumberNormalizer()
        self.standardize_spellings = DutchSpellingNormalizer(
            dutch_spelling_mapping)

    def convert_time_format(self, s: str) -> str:
        # Regular expression to match times in "HH:MM" format
        def time_replacer(match):
            hour = int(match.group(1))
            hour = hour % 12 if hour > 12 else hour
            next_hour = (hour+1) % 12 if (hour+1) > 12 else (hour+1)
            minute = int(match.group(3))
            if minute == 00:
                return f"{hour} uur"
            elif minute == 15:
                return f"kwart over {hour}"
            elif minute == 30:
                return f"half {next_hour}"
            elif minute == 45:
                return f"kwart voor {next_hour}"
            return match.group(0)

        # Replace "09:00" with "9 uur"
        s = re.sub(r'\b0?(\d{1,2})(:|\.)(00|15|30|45)\b(\s?uur)?', time_replacer, s)

        return s

    def __call__(self, s: str):
        s = s.lower()

        # remove words between brackets
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        # Convert times before applying other rules
        s = self.convert_time_format(s)
        s = re.sub(self.ignore_patterns, "", s)
        # standardize when there's a space before an apostrophe
        s = re.sub(r"\s+'", "'", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        # s = re.sub(r"(\d).(\d)", r"\1\2", s)  # remove dots between digits
        # remove periods not followed by numbers
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)
        # keep some symbols for numerics
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        # replace any successive whitespace characters with a space
        s = re.sub(r"\s+", " ", s)

        return s
