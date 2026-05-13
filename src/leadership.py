"""
Committee leadership enrichment: chair, ranking member, and leader flags.

Provides a static lookup table of House committee leadership for the 115th-118th
Congresses and functions to enrich pipeline data with leadership indicators.

Data sources:
    - Committee chairs and ranking members compiled from congress.gov committee
      pages and cross-referenced with official House records.
    - Bioguide IDs verified against BICAM members.csv.
    - Committee codes use the THOMAS ID convention (e.g. HSAG = House Agriculture).

BICAM committee codes (e.g. 'hsag00') are normalized to THOMAS IDs by uppercasing
the first 4 characters and dropping the 2-digit suffix.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Committee leadership lookup: (congress, thomas_id) -> {"chair": bioguide_id, "ranking_member": bioguide_id}
#
# Sources: congress.gov committee pages for each congress session.
# Only House standing committees with hearings in BICAM are included.
# Committee codes follow the THOMAS ID convention used by congress-legislators.
#
# Mid-congress changes:
#   - 115th HSRU: Ranking Member Louise Slaughter (S000480) died March 2018;
#     Jim McGovern (M000312) succeeded her. Both are included.
#   - 116th HSGO: Chair Elijah Cummings (C000984) died October 2019;
#     Carolyn Maloney (M000087) succeeded him. Both are included.
COMMITTEE_LEADERS: dict[tuple[int, str], dict[str, str | list[str]]] = {
    # =========================================================================
    # 115th Congress (2017-2019) — Republican majority
    # =========================================================================
    # Agriculture
    (115, "HSAG"): {"chair": "C001062", "ranking_member": "P000258"},  # Conaway / Peterson
    # Appropriations
    (115, "HSAP"): {"chair": "F000372", "ranking_member": "L000480"},  # Frelinghuysen / Lowey
    # Armed Services
    (115, "HSAS"): {"chair": "T000238", "ranking_member": "S000510"},  # Thornberry / Adam Smith
    # Budget
    (115, "HSBU"): {"chair": "B001273", "ranking_member": "Y000062"},  # Black / Yarmuth
    # Education and the Workforce
    (115, "HSED"): {"chair": "F000450", "ranking_member": "S000185"},  # Foxx / Bobby Scott
    # Energy and Commerce
    (115, "HSIF"): {"chair": "W000791", "ranking_member": "P000034"},  # Walden / Pallone
    # Ethics
    (115, "HSSO"): {"chair": "B001284", "ranking_member": "D000610"},  # Susan Brooks / Deutch
    # Financial Services
    (115, "HSBA"): {"chair": "H001036", "ranking_member": "W000187"},  # Hensarling / Waters
    # Foreign Affairs
    (115, "HSFA"): {"chair": "R000487", "ranking_member": "E000179"},  # Royce / Engel
    # Homeland Security
    (115, "HSHM"): {"chair": "M001157", "ranking_member": "T000193"},  # McCaul / Bennie Thompson
    # House Administration
    (115, "HSHA"): {"chair": "H001045", "ranking_member": "B001227"},  # Harper / Robert Brady
    # Judiciary
    (115, "HSJU"): {"chair": "G000289", "ranking_member": "N000002"},  # Goodlatte / Nadler
    # Natural Resources
    (115, "HSII"): {"chair": "B001250", "ranking_member": "G000551"},  # Rob Bishop / Grijalva
    # Oversight and Government Reform
    (115, "HSGO"): {"chair": "G000566", "ranking_member": "C000984"},  # Gowdy / Cummings
    # Rules (Slaughter died March 2018, McGovern succeeded)
    (115, "HSRU"): {"chair": "S000250", "ranking_member": ["S000480", "M000312"]},  # Pete Sessions / Slaughter→McGovern
    # Science, Space, and Technology
    (115, "HSSY"): {"chair": "S000583", "ranking_member": "J000126"},  # Lamar Smith / Eddie B. Johnson
    # Small Business
    (115, "HSSM"): {"chair": "C000266", "ranking_member": "V000081"},  # Chabot / Velázquez
    # Transportation and Infrastructure
    (115, "HSPW"): {"chair": "S001154", "ranking_member": "D000191"},  # Shuster / DeFazio
    # Veterans' Affairs
    (115, "HSVR"): {"chair": "R000582", "ranking_member": "W000799"},  # Roe / Walz
    # Ways and Means
    (115, "HSWM"): {"chair": "B000755", "ranking_member": "N000015"},  # Kevin Brady / Neal
    # =========================================================================
    # 116th Congress (2019-2021) — Democratic majority
    # =========================================================================
    # Agriculture
    (116, "HSAG"): {"chair": "P000258", "ranking_member": "C001062"},  # Peterson / Conaway
    # Appropriations
    (116, "HSAP"): {"chair": "L000480", "ranking_member": "G000377"},  # Lowey / Granger
    # Armed Services
    (116, "HSAS"): {"chair": "S000510", "ranking_member": "T000238"},  # Adam Smith / Thornberry
    # Budget
    (116, "HSBU"): {"chair": "Y000062", "ranking_member": "W000809"},  # Yarmuth / Womack
    # Education and Labor
    (116, "HSED"): {"chair": "S000185", "ranking_member": "F000450"},  # Bobby Scott / Foxx
    # Energy and Commerce
    (116, "HSIF"): {"chair": "P000034", "ranking_member": "W000791"},  # Pallone / Walden
    # Ethics
    (116, "HSSO"): {"chair": "D000610", "ranking_member": "M001158"},  # Deutch / Marchant
    # Financial Services
    (116, "HSBA"): {"chair": "W000187", "ranking_member": "M001156"},  # Waters / McHenry
    # Foreign Affairs
    (116, "HSFA"): {"chair": "E000179", "ranking_member": "M001157"},  # Engel / McCaul
    # Homeland Security
    (116, "HSHM"): {"chair": "T000193", "ranking_member": "R000575"},  # Bennie Thompson / Mike Rogers (AL)
    # House Administration
    (116, "HSHA"): {"chair": "L000397", "ranking_member": "D000619"},  # Lofgren / Rodney Davis
    # Judiciary
    (116, "HSJU"): {"chair": "N000002", "ranking_member": "C001093"},  # Nadler / Doug Collins
    # Natural Resources
    (116, "HSII"): {"chair": "G000551", "ranking_member": "B001250"},  # Grijalva / Rob Bishop
    # Oversight and Reform (Cummings died Oct 2019, Maloney succeeded)
    (116, "HSGO"): {"chair": ["C000984", "M000087"], "ranking_member": "J000289"},  # Cummings→Maloney / Jordan
    # Rules
    (116, "HSRU"): {"chair": "M000312", "ranking_member": "C001053"},  # McGovern / Cole
    # Science, Space, and Technology
    (116, "HSSY"): {"chair": "J000126", "ranking_member": "L000491"},  # Eddie B. Johnson / Lucas
    # Small Business
    (116, "HSSM"): {"chair": "V000081", "ranking_member": "C000266"},  # Velázquez / Chabot
    # Transportation and Infrastructure
    (116, "HSPW"): {"chair": "D000191", "ranking_member": "G000546"},  # DeFazio / Sam Graves
    # Veterans' Affairs
    (116, "HSVR"): {"chair": "T000472", "ranking_member": "R000582"},  # Takano / Roe
    # Ways and Means
    (116, "HSWM"): {"chair": "N000015", "ranking_member": "B000755"},  # Neal / Kevin Brady
    # =========================================================================
    # 117th Congress (2021-2023) — Democratic majority
    # =========================================================================
    # Agriculture
    (117, "HSAG"): {"chair": "S001157", "ranking_member": "T000467"},  # David Scott / Glenn Thompson
    # Appropriations
    (117, "HSAP"): {"chair": "D000216", "ranking_member": "G000377"},  # DeLauro / Granger
    # Armed Services
    (117, "HSAS"): {"chair": "S000510", "ranking_member": "R000575"},  # Adam Smith / Mike Rogers (AL)
    # Budget
    (117, "HSBU"): {"chair": "Y000062", "ranking_member": "S001195"},  # Yarmuth / Jason Smith
    # Education and Labor
    (117, "HSED"): {"chair": "S000185", "ranking_member": "F000450"},  # Bobby Scott / Foxx
    # Energy and Commerce
    (117, "HSIF"): {"chair": "P000034", "ranking_member": "M001159"},  # Pallone / McMorris Rodgers
    # Ethics
    (117, "HSSO"): {"chair": "D000610", "ranking_member": "W000813"},  # Deutch / Walorski
    # Financial Services
    (117, "HSBA"): {"chair": "W000187", "ranking_member": "M001156"},  # Waters / McHenry
    # Foreign Affairs
    (117, "HSFA"): {"chair": "M001137", "ranking_member": "M001157"},  # Meeks / McCaul
    # Homeland Security
    (117, "HSHM"): {"chair": "T000193", "ranking_member": "K000386"},  # Bennie Thompson / Katko
    # House Administration
    (117, "HSHA"): {"chair": "L000397", "ranking_member": "D000619"},  # Lofgren / Rodney Davis
    # Judiciary
    (117, "HSJU"): {"chair": "N000002", "ranking_member": "J000289"},  # Nadler / Jordan
    # Natural Resources
    (117, "HSII"): {"chair": "G000551", "ranking_member": "W000821"},  # Grijalva / Westerman
    # Oversight and Reform
    (117, "HSGO"): {"chair": "M000087", "ranking_member": "C001108"},  # Maloney / Comer
    # Rules
    (117, "HSRU"): {"chair": "M000312", "ranking_member": "C001053"},  # McGovern / Cole
    # Science, Space, and Technology
    (117, "HSSY"): {"chair": "J000126", "ranking_member": "L000491"},  # Eddie B. Johnson / Lucas
    # Small Business
    (117, "HSSM"): {"chair": "V000081", "ranking_member": "L000569"},  # Velázquez / Luetkemeyer
    # Transportation and Infrastructure
    (117, "HSPW"): {"chair": "D000191", "ranking_member": "G000546"},  # DeFazio / Sam Graves
    # Veterans' Affairs
    (117, "HSVR"): {"chair": "T000472", "ranking_member": "B001295"},  # Takano / Bost
    # Ways and Means
    (117, "HSWM"): {"chair": "N000015", "ranking_member": "B000755"},  # Neal / Kevin Brady
    # =========================================================================
    # 118th Congress (2023-2025) — Republican majority
    # =========================================================================
    # Agriculture
    (118, "HSAG"): {"chair": "T000467", "ranking_member": "S001157"},  # Glenn Thompson / David Scott
    # Appropriations
    (118, "HSAP"): {"chair": "G000377", "ranking_member": "D000216"},  # Granger / DeLauro
    # Armed Services
    (118, "HSAS"): {"chair": "R000575", "ranking_member": "S000510"},  # Mike Rogers (AL) / Adam Smith
    # Budget
    (118, "HSBU"): {"chair": "A000375", "ranking_member": "B001296"},  # Arrington / Boyle
    # Education and the Workforce
    (118, "HSED"): {"chair": "F000450", "ranking_member": "S000185"},  # Foxx / Bobby Scott
    # Energy and Commerce
    (118, "HSIF"): {"chair": "M001159", "ranking_member": "P000034"},  # McMorris Rodgers / Pallone
    # Ethics
    (118, "HSSO"): {"chair": "G000591", "ranking_member": "W000826"},  # Guest / Wild
    # Financial Services
    (118, "HSBA"): {"chair": "M001156", "ranking_member": "W000187"},  # McHenry / Waters
    # Foreign Affairs
    (118, "HSFA"): {"chair": "M001157", "ranking_member": "M001137"},  # McCaul / Meeks
    # Homeland Security
    (118, "HSHM"): {"chair": "G000590", "ranking_member": "T000193"},  # Mark Green / Bennie Thompson
    # House Administration
    (118, "HSHA"): {"chair": "L000583", "ranking_member": "M001206"},  # Loudermilk / Morelle
    # Judiciary
    (118, "HSJU"): {"chair": "J000289", "ranking_member": "N000002"},  # Jordan / Nadler
    # Natural Resources
    (118, "HSII"): {"chair": "W000821", "ranking_member": "G000551"},  # Westerman / Grijalva
    # Oversight and Accountability
    (118, "HSGO"): {"chair": "C001108", "ranking_member": "R000606"},  # Comer / Raskin
    # Rules
    (118, "HSRU"): {"chair": "C001053", "ranking_member": "M000312"},  # Cole / McGovern
    # Science, Space, and Technology
    (118, "HSSY"): {"chair": "L000491", "ranking_member": "L000397"},  # Lucas / Lofgren
    # Small Business
    (118, "HSSM"): {"chair": "W000816", "ranking_member": "V000081"},  # Roger Williams / Velázquez
    # Transportation and Infrastructure
    (118, "HSPW"): {"chair": "G000546", "ranking_member": "L000560"},  # Sam Graves / Larsen
    # Veterans' Affairs
    (118, "HSVR"): {"chair": "B001295", "ranking_member": "T000472"},  # Bost / Takano
    # Ways and Means
    (118, "HSWM"): {"chair": "S001195", "ranking_member": "N000015"},  # Jason Smith / Neal
}


def normalize_committee_code(bicam_code: str) -> str:
    """
    Convert a BICAM committee code to a THOMAS ID.

    BICAM codes are lowercase with a 2-digit suffix (e.g. 'hsag00' for the
    parent House Agriculture committee). The THOMAS ID equivalent is the
    first 4 characters uppercased (e.g. 'HSAG').

    Only parent committee codes (suffix '00') are meaningful for leadership
    lookups. Subcommittee codes (suffix != '00') return empty string.

    Args:
        bicam_code: BICAM-style committee code (e.g. 'hsag00').

    Returns:
        THOMAS ID string (e.g. 'HSAG'), or empty string if the code is
        invalid or represents a subcommittee.
    """
    if not isinstance(bicam_code, str) or len(bicam_code) < 6:
        return ""
    suffix = bicam_code[4:]
    if suffix != "00":
        return ""  # Subcommittee — no leadership lookup
    return bicam_code[:4].upper()


def load_committee_leaders(target_congresses=None):
    """
    Build a DataFrame of committee leaders for the target congresses.

    Each row represents one leader assignment: a bioguide_id holding a role
    (chair or ranking_member) on a committee in a given congress.

    Args:
        target_congresses: List of congress numbers to include. If None,
                          returns all congresses in COMMITTEE_LEADERS.

    Returns:
        DataFrame with columns: bioguide_id, congress, thomas_id, role
        where role is 'chair' or 'ranking_member'.
    """
    rows = []
    for (congress, thomas_id), leaders in COMMITTEE_LEADERS.items():
        if target_congresses is not None and congress not in target_congresses:
            continue
        for role in ("chair", "ranking_member"):
            bio_ids = leaders[role]
            if isinstance(bio_ids, list):
                for bio_id in bio_ids:
                    rows.append(
                        {
                            "bioguide_id": bio_id,
                            "congress": congress,
                            "thomas_id": thomas_id,
                            "role": role,
                        }
                    )
            else:
                rows.append(
                    {
                        "bioguide_id": bio_ids,
                        "congress": congress,
                        "thomas_id": thomas_id,
                        "role": role,
                    }
                )

    df = pd.DataFrame(rows, columns=["bioguide_id", "congress", "thomas_id", "role"])
    if target_congresses is not None:
        logger.info(
            "Loaded committee leaders for congresses %s: %d entries (%d chairs, %d ranking members)",
            target_congresses,
            len(df),
            (df["role"] == "chair").sum(),
            (df["role"] == "ranking_member").sum(),
        )
    return df


def prepare_leadership_enrichment(df):
    """
    Enrich a pipeline DataFrame with committee leadership indicators.

    For each row, checks whether the speaker (identified by bioguide_id) holds
    a leadership role on the hearing's committee (identified by committee_code)
    in the relevant congress.

    Adds three columns:
        - chairspeech (int, 0/1): 1 if the speaker is the committee chair
        - rankmemspeech (int, 0/1): 1 if the speaker is the ranking member
        - leader (int, 0/1): 1 if either chair or ranking member

    Args:
        df: Pipeline DataFrame. Must contain columns: bioguide_id, congress,
            committee_code. Rows with missing bioguide_id get 0 for all flags.

    Returns:
        DataFrame with the three new columns added.
    """
    if df.empty:
        df["chairspeech"] = pd.Series(dtype=int)
        df["rankmemspeech"] = pd.Series(dtype=int)
        df["leader"] = pd.Series(dtype=int)
        return df

    # Normalize committee codes to THOMAS IDs
    df = df.copy()
    df["_thomas_id"] = df["committee_code"].apply(normalize_committee_code)

    # Load leaders for the congresses present in the data
    target_congresses = sorted(df["congress"].dropna().unique().astype(int).tolist())
    leaders_df = load_committee_leaders(target_congresses=target_congresses)

    # Build separate lookup sets for chairs and ranking members
    chair_df = leaders_df[leaders_df["role"] == "chair"][["bioguide_id", "congress", "thomas_id"]]
    rm_df = leaders_df[leaders_df["role"] == "ranking_member"][["bioguide_id", "congress", "thomas_id"]]

    # Merge to flag chairs
    df = df.merge(
        chair_df.assign(_is_chair=1),
        left_on=["bioguide_id", "congress", "_thomas_id"],
        right_on=["bioguide_id", "congress", "thomas_id"],
        how="left",
    )
    df["chairspeech"] = df["_is_chair"].fillna(0).astype(int)
    df = df.drop(columns=["_is_chair", "thomas_id"], errors="ignore")

    # Merge to flag ranking members
    df = df.merge(
        rm_df.assign(_is_rm=1),
        left_on=["bioguide_id", "congress", "_thomas_id"],
        right_on=["bioguide_id", "congress", "thomas_id"],
        how="left",
    )
    df["rankmemspeech"] = df["_is_rm"].fillna(0).astype(int)
    df = df.drop(columns=["_is_rm", "thomas_id", "_thomas_id"], errors="ignore")

    # Derived: leader = chair OR ranking member
    df["leader"] = ((df["chairspeech"] == 1) | (df["rankmemspeech"] == 1)).astype(int)

    # Log summary
    n_chair = (df["chairspeech"] == 1).sum()
    n_rm = (df["rankmemspeech"] == 1).sum()
    n_leader = (df["leader"] == 1).sum()
    n_total = len(df)
    logger.info(
        "Leadership enrichment: %d chairs (%.1f%%), %d ranking members (%.1f%%), %d leaders total (%.1f%%) out of %d rows",
        n_chair,
        n_chair / n_total * 100 if n_total > 0 else 0,
        n_rm,
        n_rm / n_total * 100 if n_total > 0 else 0,
        n_leader,
        n_leader / n_total * 100 if n_total > 0 else 0,
        n_total,
    )

    return df
