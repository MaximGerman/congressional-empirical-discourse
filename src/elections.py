import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "external")
DEST_PATH = os.path.join(DATA_DIR, "1976-2022-house.csv")
FILE_ID = "13592823"  # Dataverse file ID for the MIT House election returns


def fetch_mit_election_data(dest_path=DEST_PATH):
    """
    Fetches the MIT Election Lab House dataset from Harvard Dataverse.
    Requires DATAVERSE_API_TOKEN in .env.
    """
    if os.path.exists(dest_path):
        logger.info("MIT Election data already exists at %s", dest_path)
        return dest_path

    load_dotenv()
    api_token = os.getenv("DATAVERSE_API_TOKEN")

    url = f"https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}?format=original"
    headers = {}
    if api_token:
        headers["X-Dataverse-key"] = api_token

    logger.info("Downloading MIT Election data from Dataverse...")
    # First attempt: standard GET
    response = requests.get(url, headers=headers, stream=True)

    # If it fails with a guestbook error (400), try a POST to get a signed URL
    if response.status_code == 400 and "Guestbook" in response.text:
        logger.info("Guestbook required. Attempting to obtain signed URL...")
        guestbook_data = {
            "name": "Maxim German",
            "email": "maximgerman1@mail.tau.ac.il",
            "institution": "Tel Aviv University",
            "position": "Student",
        }
        post_response = requests.post(url, headers=headers, json=guestbook_data)
        if post_response.status_code == 200:
            json_data = post_response.json()
            if "data" in json_data and "signedUrl" in json_data["data"]:
                signed_url = json_data["data"]["signedUrl"]
                logger.info("Obtained signed URL. Retrying download...")
                response = requests.get(signed_url, stream=True)
            else:
                logger.error("Signed URL not found in response: %s", json_data)
        else:
            logger.error(
                "Failed to obtain signed URL. Status: %d, Body: %s", post_response.status_code, post_response.text
            )

    if response.status_code == 200:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Successfully downloaded MIT Election data to %s", dest_path)
        return dest_path
    else:
        error_msg = f"Failed to download MIT Election data. Status: {response.status_code}"
        if "Guestbook" in response.text:
            error_msg += " (Guestbook required. Please check your DATAVERSE_API_TOKEN)"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def load_elections_data(path=None, target_congresses=None):
    """
    Loads and processes MIT Election Lab data to calculate winning vote share.
    """
    if path is None:
        path = fetch_mit_election_data()

    if target_congresses is None:
        target_congresses = [115, 116, 117, 118]

    logger.info("Loading MIT Election data from %s", path)

    # Read the CSV. The MIT data uses some special encodings occasionally, but mostly utf-8.
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)

    # Map election year to congress: year 2016 -> 115th congress
    df["congress"] = (df["year"] - 1788) // 2 + 1

    # Filter to relevant congresses
    df = df[df["congress"].isin(target_congresses)].copy()

    # Calculate vote percentage for each candidate
    # Handle cases where totalvotes might be 0 to avoid division by zero
    df["vote_pct"] = (df["candidatevotes"] / df["totalvotes"].replace(0, pd.NA)) * 100
    pd.set_option("future.no_silent_downcasting", True)
    df["vote_pct"] = df["vote_pct"].fillna(100.0).infer_objects()  # Unopposed where totalvotes == 0

    # Ensure district is an integer (MIT data has district 0 for at-large)
    df["district_code"] = pd.to_numeric(df["district"], errors="coerce").fillna(0).astype(int)

    # We want the winning candidate's vote share for that district
    # Group by state_po, district, congress and take the max vote_pct
    winning_shares = df.groupby(["state_po", "district_code", "congress"])["vote_pct"].max().reset_index()

    # Rename for merging with Voteview
    winning_shares.rename(columns={"state_po": "state_abbrev"}, inplace=True)

    # Calculate quadratic term
    winning_shares["vote_pct_sq"] = winning_shares["vote_pct"] ** 2

    logger.info(
        "Processed election data for %d districts across %d congresses",
        len(winning_shares),
        len(winning_shares["congress"].unique()),
    )

    return winning_shares
