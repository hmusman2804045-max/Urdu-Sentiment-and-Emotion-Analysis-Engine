import re

URDU_SCRIPT_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
LATIN_PATTERN = re.compile(r'[a-zA-Z]')

ROMAN_URDU_KEYWORDS = {
    'hai', 'hain', 'tha', 'thi', 'thay', 'the', 'main', 'mai', 'mujhe', 'mujhy',
    'tum', 'tumhein', 'aap', 'ap', 'bohat', 'buhat', 'achha', 'acha', 'achi',
    'kya', 'kia', 'kyun', 'kyu', 'ko', 'se', 'par', 'pe', 'ne', 'gaya', 'gaye',
    'gayi', 'huye', 'hue', 'wala', 'wali', 'wale', 'raha', 'rahi', 'rahe',
    'sirf', 'bhi', 'magar', 'lekin', 'lekan', 'aur', 'or', 'ka', 'ki', 'ke',
    'nhi', 'nahi', 'nahin', 'ya', 'jo', 'jab', 'tab', 'ab', 'sab', 'zaroor',
    'phir', 'bhi', 'ye', 'yeh', 'wo', 'woh', 'karna', 'karo', 'karein', 'karta'
}

ENGLISH_COMMON_KEYWORDS = {
    'the', 'is', 'are', 'was', 'were', 'this', 'that', 'with', 'from', 'have',
    'has', 'had', 'not', 'but', 'what', 'where', 'when', 'who', 'how', 'very',
    'good', 'bad', 'great', 'awesome', 'terrible', 'happy', 'sad', 'angry'
}

def detect_language(text: str) -> str:
    if not text or not str(text).strip():
        return "Unknown"

    text_str = str(text).strip()
    has_urdu_script = bool(URDU_SCRIPT_PATTERN.search(text_str))
    has_latin = bool(LATIN_PATTERN.search(text_str))

    if has_urdu_script and has_latin:
        return "Mixed"
    
    if has_urdu_script and not has_latin:
        return "Urdu Script"

    if has_latin and not has_urdu_script:
        words = re.findall(r'\b[a-zA-Z]+\b', text_str.lower())
        if not words:
            return "English"

        roman_score = sum(1 for w in words if w in ROMAN_URDU_KEYWORDS)
        english_score = sum(1 for w in words if w in ENGLISH_COMMON_KEYWORDS)

        if roman_score > 0 and english_score > 0:
            return "Mixed"
        elif roman_score > 0:
            return "Roman Urdu"
        elif english_score > 0:
            return "English"
        else:
            # Heuristic for unknown latin words: default to Roman Urdu if common length/structure
            return "Roman Urdu"

    return "Unknown"
