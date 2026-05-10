import pandas as pd
import pytest

from src.pipeline import step3_process_transcripts, step5_create_sample


@pytest.fixture
def sample_transcript():
    """A minimal transcript that exercises speaker segmentation and sentence splitting."""
    return (
        "    Mr. Smith. Thank you, Mr. Chairman. "
        "I appreciate the opportunity to speak today.\n"
        "    Mrs. Jones. I yield my time to the gentleman from California.\n"
        "    Dr. Fauci. Thank you for having me. "
        "The data shows a clear trend.\n"
    )


@pytest.fixture
def new_era_df():
    return pd.DataFrame(
        {
            "hearing_id": [1, 2],
            "congress": [115, 116],
            "chamber": ["house", "house"],
        }
    )


@pytest.fixture
def texts_df(sample_transcript):
    return pd.DataFrame(
        {
            "hearing_id": [1, 2],
            "raw_text": [sample_transcript, sample_transcript],
        }
    )


class TestStep3ProcessTranscripts:
    def test_extracts_sentences(self, new_era_df, texts_df):
        sentences_df, failed = step3_process_transcripts(new_era_df, texts_df)

        assert failed == []
        assert not sentences_df.empty
        assert "hearing_id" in sentences_df.columns
        assert "speaker" in sentences_df.columns
        assert "target_sentence" in sentences_df.columns

    def test_multiple_hearings(self, new_era_df, texts_df):
        sentences_df, _ = step3_process_transcripts(new_era_df, texts_df)

        hearing_ids = sentences_df["hearing_id"].unique()
        assert len(hearing_ids) == 2
        assert set(hearing_ids) == {1, 2}

    def test_empty_text_skipped(self, new_era_df):
        texts_df = pd.DataFrame(
            {
                "hearing_id": [1],
                "raw_text": ["short"],  # below 100-char threshold
            }
        )
        sentences_df, failed = step3_process_transcripts(new_era_df, texts_df)

        assert sentences_df.empty
        assert failed == []

    def test_includes_witness_speakers(self, new_era_df, texts_df):
        """step3 does NOT filter witnesses — that happens in step4."""
        sentences_df, _ = step3_process_transcripts(new_era_df, texts_df)

        speakers = sentences_df["speaker"].unique()
        # Dr. Fauci should be present (witness filtering is step4's job)
        assert any("Fauci" in s for s in speakers)


class TestStep5CreateSample:
    @pytest.fixture
    def legislators_df(self):
        """Simulates enriched legislator data across two congresses."""
        rows = []
        for congress in [115, 116]:
            for i in range(200):
                rows.append(
                    {
                        "hearing_id": i,
                        "speaker": f"Mr. SPEAKER{i}",
                        "target_sentence": f"Sentence {i}.",
                        "congress": congress,
                        "party": "Republican" if i % 2 == 0 else "Democratic",
                        "bioguide_id": f"X{congress:03d}{i:04d}",
                        "minority": i % 2,
                    }
                )
        return pd.DataFrame(rows)

    def test_sample_size_respected(self, legislators_df):
        sample = step5_create_sample(legislators_df, sample_size=50, seed=42)
        assert len(sample) == 50

    def test_stratified_by_congress(self, legislators_df):
        sample = step5_create_sample(legislators_df, sample_size=100, seed=42)

        congress_counts = sample["congress"].value_counts()
        # Each congress should get roughly half
        assert congress_counts[115] == 50
        assert congress_counts[116] == 50

    def test_small_dataset_uses_all(self):
        small_df = pd.DataFrame(
            {
                "hearing_id": [1, 2, 3],
                "speaker": ["Mr. A", "Mr. B", "Mr. C"],
                "target_sentence": ["s1", "s2", "s3"],
                "congress": [115, 115, 116],
                "party": ["Republican", "Democratic", "Republican"],
                "bioguide_id": ["X001", "X002", "X003"],
                "minority": [0, 1, 0],
            }
        )
        sample = step5_create_sample(small_df, sample_size=10000, seed=42)
        assert len(sample) == 3

    def test_empty_input(self):
        sample = step5_create_sample(pd.DataFrame(), sample_size=100)
        assert sample.empty

    def test_unmatched_rows_excluded(self):
        df = pd.DataFrame(
            {
                "hearing_id": [1, 2],
                "speaker": ["Mr. A", "Mr. B"],
                "target_sentence": ["s1", "s2"],
                "congress": [115, 115],
                "party": ["Republican", None],
                "bioguide_id": ["X001", None],  # second row has no match
                "minority": [0, None],
            }
        )
        sample = step5_create_sample(df, sample_size=100, seed=42)
        assert len(sample) == 1
        assert sample.iloc[0]["bioguide_id"] == "X001"
