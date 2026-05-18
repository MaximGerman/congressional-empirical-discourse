import pandas as pd
import pytest

from src.preprocess import (
    create_sentence_records,
    extract_last_name,
    extract_name_parts,
    is_likely_witness,
    match_speaker_to_member,
    segment_speakers,
)


def test_segment_speakers_basic():
    text = """
    Mr. SMITH. Thank you for having me.
This is a test.
    Mrs. JONES. I agree.
    """
    chunks = segment_speakers(text)
    assert len(chunks) == 2
    assert chunks[0]["speaker"] == "Mr. SMITH"
    assert chunks[0]["text"] == "Thank you for having me. This is a test."
    assert chunks[1]["speaker"] == "Mrs. JONES"
    assert chunks[1]["text"] == "I agree."


def test_segment_speakers_chair():
    text = """
    The CHAIRMAN. We will begin.
    The Chairwoman. Please sit down.
    """
    chunks = segment_speakers(text)
    assert len(chunks) == 2
    assert chunks[0]["speaker"] == "The CHAIRMAN"
    assert chunks[0]["text"] == "We will begin."
    assert chunks[1]["speaker"] == "The Chairwoman"
    assert chunks[1]["text"] == "Please sit down."


def test_segment_speakers_empty():
    assert segment_speakers("") == []
    assert segment_speakers(None) == []


def test_segment_speakers_toc_filtering():
    text = """
    Mr. SMITH. Hello.
Table of contents .... 5
    Mr. JONES. Goodbye.
    """
    chunks = segment_speakers(text)
    assert len(chunks) == 2
    assert chunks[0]["speaker"] == "Mr. SMITH"
    assert chunks[0]["text"] == "Hello."
    assert chunks[1]["speaker"] == "Mr. JONES"
    assert chunks[1]["text"] == "Goodbye."


def test_segment_speakers_three_word_name():
    """Three-word compound names (e.g., De La Cruz) must be captured after the regex fix."""
    text = "    Representative De La Cruz. Thank you, Mr. Chairman.\n"
    chunks = segment_speakers(text)
    assert len(chunks) == 1
    assert chunks[0]["speaker"] == "Representative De La Cruz"


def test_segment_speakers_three_word_name_van_de():
    text = "    Mr. Van De Putte. I yield my time.\n"
    chunks = segment_speakers(text)
    assert len(chunks) == 1
    assert chunks[0]["speaker"] == "Mr. Van De Putte"


def test_extract_name_parts_single_word():
    full, last = extract_name_parts("Mr. SMITH")
    assert full == "SMITH"
    assert last == "SMITH"


def test_extract_name_parts_multi_word():
    full, last = extract_name_parts("Ms. Jackson Lee")
    assert full == "JACKSON LEE"
    assert last == "LEE"


def test_extract_name_parts_van_prefix():
    full, last = extract_name_parts("Mr. Van Orden")
    assert full == "VAN ORDEN"
    assert last == "ORDEN"


def test_extract_name_parts_three_word():
    full, last = extract_name_parts("Representative De La Cruz")
    assert full == "DE LA CRUZ"
    assert last == "CRUZ"


def test_extract_name_parts_chair():
    full, last = extract_name_parts("The Chairman")
    assert full == "CHAIRMAN"
    assert last == "CHAIRMAN"


def test_extract_name_parts_chairman_with_name():
    full, last = extract_name_parts("Chairman SMITH")
    assert full == "SMITH"
    assert last == "SMITH"


def test_extract_name_parts_empty():
    full, last = extract_name_parts("")
    assert full is None
    assert last is None
    full, last = extract_name_parts(None)
    assert full is None
    assert last is None


def test_extract_last_name():
    assert extract_last_name("Mr. SMITH") == "SMITH"
    assert extract_last_name("Mrs. JONES.") == "JONES"
    assert extract_last_name("The CHAIRMAN") == "CHAIRMAN"
    # Multi-word names now return the full name
    assert extract_last_name("Ms. Jackson Lee") == "JACKSON LEE"
    assert extract_last_name("Mr. Van Orden") == "VAN ORDEN"
    assert extract_last_name("Ms. Wasserman Schultz") == "WASSERMAN SCHULTZ"


def test_extract_last_name_empty():
    assert extract_last_name("") is None
    assert extract_last_name(None) is None


def test_create_sentence_records():
    # Mocking sent_tokenize isn't strictly necessary as we download the punkt_tab model
    # but we should ensure it's available.
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    chunks = [{"speaker": "Mr. SMITH", "text": "Sentence one. Sentence two. Sentence three."}]
    records = create_sentence_records(chunks, hearing_id=123)

    assert len(records) == 3
    assert records[0]["target_sentence"] == "Sentence one."
    assert records[0]["context_before"] == ""
    assert records[0]["context_after"] == "Sentence two."
    assert records[0]["speaker_last_name"] == "SMITH"
    assert records[0]["speaker_last_word"] == "SMITH"

    assert records[1]["target_sentence"] == "Sentence two."
    assert records[1]["context_before"] == "Sentence one."
    assert records[1]["context_after"] == "Sentence three."


def test_create_sentence_records_multi_word_name():
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    chunks = [{"speaker": "Ms. Jackson Lee", "text": "Thank you, Mr. Chairman."}]
    records = create_sentence_records(chunks, hearing_id=456)

    assert len(records) == 1
    assert records[0]["speaker_last_name"] == "JACKSON LEE"
    assert records[0]["speaker_last_word"] == "LEE"


def test_create_sentence_records_short_filter():
    chunks = [{"speaker": "Mr. SMITH", "text": "Sentence one. Ok. Sentence three."}]
    records = create_sentence_records(chunks, hearing_id=123)
    assert len(records) == 2
    assert records[0]["target_sentence"] == "Sentence one."
    assert records[1]["target_sentence"] == "Sentence three."


@pytest.fixture
def member_lookup():
    return pd.DataFrame(
        {
            "bioguide_id": ["A000001", "B000002", "C000003", "D000004"],
            "last_name": ["Smith", "Smythe", "Jones", "Jackson Lee"],
            "first_name": ["John", "Jane", "Mary", "Sheila"],
            "party": ["Republican", "Democratic", "Democratic", "Democratic"],
            "state": ["CA", "NY", "TX", "TX"],
            "congress": [115, 115, 116, 115],
            "last_name_upper": ["SMITH", "SMYTHE", "JONES", "JACKSON LEE"],
        }
    )


def test_match_speaker_exact(member_lookup):
    res = match_speaker_to_member("SMITH", member_lookup, congress=115)
    assert res is not None
    assert res["bioguide_id"] == "A000001"
    assert res["match_type"] == "exact"
    assert res["match_score"] == 100


def test_match_speaker_multi_word_exact(member_lookup):
    res = match_speaker_to_member("JACKSON LEE", member_lookup, congress=115)
    assert res is not None
    assert res["bioguide_id"] == "D000004"
    assert res["match_type"] == "exact"


def test_match_speaker_last_word_fallback(member_lookup):
    # If full name doesn't match, try last word only
    res = match_speaker_to_member("JACKSON LEE", member_lookup, congress=115, speaker_last_word="LEE")
    assert res is not None
    assert res["bioguide_id"] == "D000004"


def test_match_speaker_fuzzy(member_lookup):
    # SMYTH should fuzzy match to SMYTHE
    res = match_speaker_to_member("SMYTH", member_lookup, congress=115, score_threshold=80)
    assert res is not None
    assert res["bioguide_id"] == "B000002"
    assert res["match_type"] == "fuzzy"


def test_match_speaker_ambiguous(member_lookup):
    # Add a duplicate SMITH
    ambiguous_lookup = pd.concat(
        [
            member_lookup,
            pd.DataFrame(
                {
                    "bioguide_id": ["E000005"],
                    "last_name": ["Smith"],
                    "first_name": ["Bob"],
                    "party": ["Republican"],
                    "state": ["AZ"],
                    "congress": [115],
                    "last_name_upper": ["SMITH"],
                }
            ),
        ]
    )
    res = match_speaker_to_member("SMITH", ambiguous_lookup, congress=115)
    assert res is None


def test_match_speaker_chairman(member_lookup):
    assert match_speaker_to_member("CHAIRMAN", member_lookup, congress=115) is None


def test_is_likely_witness():
    assert is_likely_witness("Mr. Smith") is False
    assert is_likely_witness("Senator Jones") is False
    assert is_likely_witness("The Chairman") is False
    assert is_likely_witness("Dr. Fauci") is True
    assert is_likely_witness("General Milley") is True
    assert is_likely_witness("John Doe") is True  # Unknown format
