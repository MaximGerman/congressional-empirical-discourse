import re

import nltk
from rapidfuzz import fuzz, process


def download_nltk_deps():
    nltk.download("punkt_tab", quiet=True)


# Speaker pattern matching actual BICAM GPO transcript format.
# Examples from real data:
#   "    Chairman Rokita. Good morning..."
#   "    Mr. Polis. Thank you."
#   "    Mrs. Davis. I appreciate..."
#   "    Ms. Fudge. Thank you, Mr. Chairman."
#   "    The Chairman. The committee will come to order."
SPEAKER_PATTERN = re.compile(
    r"^\s{2,}((?:Mr|Mrs|Ms|Dr|Chairman|Chairwoman|Chairperson|"
    r"Senator|Representative|General|Admiral|Secretary|Judge|"
    r"Ambassador|Governor|Mayor|Professor|Reverend|Father)"
    r"\.?\s+[A-Z][A-Za-z\'\-]+(?:\s+[A-Za-z\'\-]+)*)\."
)

# Pattern for "The Chairman." / "The Chairwoman." without a last name
THE_CHAIR_PATTERN = re.compile(r"^\s{2,}(The\s+Chair(?:man|woman))\.", re.IGNORECASE)


def segment_speakers(transcript_text):
    """
    Split a raw hearing transcript into per-speaker chunks.

    Handles common congressional transcript formats:
    - "Mr. SMITH. Thank you..."
    - "Chairman JONES. I would like to..."
    - "The CHAIRMAN. We will now proceed..."
    """
    if not transcript_text or not isinstance(transcript_text, str):
        return []

    chunks = []
    current_speaker = None
    current_text = []

    for line in transcript_text.split("\n"):
        # Match against the original line (preserving leading whitespace)
        # but skip table-of-contents lines (contain "...." sequences)
        if "...." in line:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        match = SPEAKER_PATTERN.match(line)
        if not match:
            match = THE_CHAIR_PATTERN.match(line)
        if match:
            # Save previous speaker's chunk
            if current_speaker and current_text:
                chunks.append({"speaker": current_speaker, "text": " ".join(current_text)})
            current_speaker = match.group(1).strip()
            # Everything after the match end (which is at the ".") is the speech
            full_match_end = match.end(0)  # end of full match including the "."
            remainder = line[full_match_end:].strip()
            current_text = [remainder] if remainder else []
        else:
            if current_speaker and stripped:
                current_text.append(stripped)

    # Don't forget the last speaker
    if current_speaker and current_text:
        chunks.append({"speaker": current_speaker, "text": " ".join(current_text)})

    return chunks


TITLE_WORDS = {
    "MR",
    "MR.",
    "MRS",
    "MRS.",
    "MS",
    "MS.",
    "DR",
    "DR.",
    "CHAIRMAN",
    "CHAIRWOMAN",
    "CHAIRPERSON",
    "SENATOR",
    "REPRESENTATIVE",
    "GENERAL",
    "ADMIRAL",
    "SECRETARY",
    "JUDGE",
    "AMBASSADOR",
    "GOVERNOR",
    "MAYOR",
    "PROFESSOR",
    "REVEREND",
    "FATHER",
    "THE",
}


def extract_name_parts(speaker_str):
    """
    Extract the full name portion from a speaker string, stripping title words.

    Returns (full_name_upper, last_word_upper) where full_name is e.g. "JACKSON LEE"
    and last_word is "LEE". For single-word names they are the same.

    Examples:
        "Mr. SMITH"           -> ("SMITH", "SMITH")
        "Ms. Jackson Lee"     -> ("JACKSON LEE", "LEE")
        "Mr. Van Orden"       -> ("VAN ORDEN", "ORDEN")
        "The Chairman"        -> ("CHAIRMAN", "CHAIRMAN")
        "Chairman SMITH"      -> ("SMITH", "SMITH")
    """
    if not speaker_str:
        return None, None
    parts = speaker_str.strip().split()
    if not parts:
        return None, None
    # Strip title words from the front
    name_parts = []
    for p in parts:
        if p.strip(".").upper() in TITLE_WORDS:
            continue
        name_parts.append(p.strip(".").upper())
    if not name_parts:
        # All tokens were titles (e.g. "The Chairman")
        last_word = parts[-1].strip(".").upper()
        return last_word, last_word
    full_name = " ".join(name_parts)
    last_word = name_parts[-1]
    return full_name, last_word


def extract_last_name(speaker_str):
    """
    Extract the last name from a speaker string like 'Mr. SMITH' or 'Chairman JONES'.
    Returns uppercase last name (full multi-word name, e.g. "JACKSON LEE").
    """
    full_name, _last_word = extract_name_parts(speaker_str)
    return full_name


def create_sentence_records(speech_chunks, hearing_id):
    """
    Split speech chunks into individual sentences with context windows.
    Each sentence gets the preceding and following sentence as context.
    """
    from nltk.tokenize import sent_tokenize

    records = []
    for chunk in speech_chunks:
        text = chunk["text"].strip()
        if not text:
            continue
        sentences = sent_tokenize(text)
        for i, sent in enumerate(sentences):
            # Skip very short sentences (likely parsing artifacts)
            if len(sent.strip()) < 5:
                continue
            full_name, last_word = extract_name_parts(chunk["speaker"])
            record = {
                "hearing_id": hearing_id,
                "speaker": chunk["speaker"],
                "speaker_last_name": full_name,
                "speaker_last_word": last_word,
                "context_before": sentences[i - 1] if i > 0 else "",
                "target_sentence": sent,
                "context_after": sentences[i + 1] if i < len(sentences) - 1 else "",
            }
            records.append(record)
    return records


def process_single_hearing(hearing_id, transcript_text):
    """
    Process a single hearing transcript end-to-end:
    raw text -> speaker chunks -> sentence records.
    """
    chunks = segment_speakers(transcript_text)
    if not chunks:
        return []
    return create_sentence_records(chunks, hearing_id)


def _try_exact_match(name, congress_members, full_speaker_name=None):
    """Try exact last_name_upper match; return member dict or None."""
    exact = congress_members[congress_members["last_name_upper"] == name]
    if len(exact) == 1:
        row = exact.iloc[0]
        return {
            "bioguide_id": row["bioguide_id"],
            "matched_name": row["last_name"],
            "first_name": row["first_name"],
            "party": row["party"],
            "state": row.get("state", ""),
            "match_score": 100,
            "match_type": "exact",
        }
    elif len(exact) > 1 and full_speaker_name:
        # Ambiguous match: Try to disambiguate using first name
        valid_candidates = []
        for _, row in exact.iterrows():
            first = str(row.get("first_name", "")).strip().upper()
            if first and (first in full_speaker_name.split() or first[0] in full_speaker_name.split()):
                valid_candidates.append(row)

        if len(valid_candidates) == 1:
            row = valid_candidates[0]
            return {
                "bioguide_id": row["bioguide_id"],
                "matched_name": row["last_name"],
                "first_name": row["first_name"],
                "party": row["party"],
                "state": row.get("state", ""),
                "match_score": 100,
                "match_type": "exact_disambiguated",
            }
    return None


def match_speaker_to_member(speaker_last_name, member_lookup_df, congress, score_threshold=85, speaker_last_word=None):
    """
    Match a speaker's last name to a known member of Congress.

    Tries in order:
    1. Exact match on full name (e.g. "JACKSON LEE")
    2. Exact match on last word only (e.g. "LEE") — for multi-word names
    3. Fuzzy match on full name

    Args:
        speaker_last_name: Uppercase full name from transcript (e.g. "JACKSON LEE")
        member_lookup_df: DataFrame from build_member_lookup()
        congress: Congress number for filtering
        score_threshold: Minimum fuzzy match score (0-100)
        speaker_last_word: Uppercase last word only (e.g. "LEE"), used as fallback

    Returns:
        dict with member info or None if no match found
    """
    if not speaker_last_name or speaker_last_name in ("CHAIRMAN", "CHAIRWOMAN", "CHAIRPERSON"):
        return None

    # Filter to members serving in this congress
    congress_members = member_lookup_df[member_lookup_df["congress"] == congress]
    if congress_members.empty:
        return None

    # Try exact match on full name first
    retVal = _try_exact_match(speaker_last_name, congress_members, full_speaker_name=speaker_last_name)
    if retVal:
        return retVal

    # For multi-word names, try exact match on last word only
    if speaker_last_word and speaker_last_word != speaker_last_name:
        retVal = _try_exact_match(speaker_last_word, congress_members, full_speaker_name=speaker_last_name)
        if retVal:
            if retVal.get("match_type") != "exact_disambiguated":
                retVal["match_type"] = "exact_last_word"
            return retVal

    # Fuzzy match
    candidates = congress_members["last_name_upper"].tolist()
    if not candidates:
        return None

    result = process.extractOne(speaker_last_name, candidates, scorer=fuzz.ratio)
    if result is None:
        return None

    matched_name, score, _idx = result
    if score < score_threshold:
        return None

    matches = congress_members[congress_members["last_name_upper"] == matched_name]
    if len(matches) != 1:
        return None

    row = matches.iloc[0]
    return {
        "bioguide_id": row["bioguide_id"],
        "matched_name": row["last_name"],
        "first_name": row["first_name"],
        "party": row["party"],
        "state": row.get("state", ""),
        "match_score": score,
        "match_type": "fuzzy",
    }


def is_likely_witness(speaker_str):
    """
    Heuristic to identify witnesses (non-legislators).
    Witnesses are typically identified by 'Dr.', professional titles,
    or they simply don't match any known member.
    """
    if not speaker_str:
        return True
    # These titles are almost always legislators
    legislator_titles = (
        "Mr.",
        "Mrs.",
        "Ms.",
        "Chairman",
        "Chairwoman",
        "The Chairman",
        "The Chairwoman",
        "Senator",
        "Representative",
    )
    return all(not speaker_str.startswith(title) for title in legislator_titles)
