import os
import re
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set, Iterable

import requests
import pandas as pd
from requests.exceptions import ConnectionError, Timeout
from urllib3.exceptions import NameResolutionError


# =========================
# USER SETTINGS
# =========================
# Choose years to fetch (edit this!)
YEARS = list(range(1995, 2026))  # 1995–2025 inclusive

# Where to write month checkpoints
OUT_DIR = "data/congressgov_committee_meetings_monthly_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# Final combined output
FINAL_CSV = "congressgov_HEARINGS_from_committee_meetings_1995_2025.csv"

# API config
BASE = "https://api.congress.gov/v3"
LIMIT = 250
SLEEP_BETWEEN_CALLS_SEC = 0.12

MAX_TRIES = 12
BACKOFF_CAP_SEC = 120

# Congress range covering 1995–2025
CONGRESS_START = 104  # starts 1995
CONGRESS_END = 119    # starts 2025

# committee-meeting chambers to try
MEETING_CHAMBERS = ["house", "senate", "nochamber"]


# =========================
# API KEY
# =========================
def load_api_key(path: str = "congress_api_key.txt") -> str:
    with open(path, "r") as f:
        key = f.read().strip()
    if not key:
        raise RuntimeError(f"API key file is empty: {path}")
    return key

API_KEY = load_api_key("congress_api_key.txt")


# =========================
# HTTP helpers
# =========================
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s

session = make_session()


def _sleep_polite():
    time.sleep(SLEEP_BETWEEN_CALLS_SEC)


def get_json(url: str, params: Optional[dict] = None, max_tries: int = MAX_TRIES) -> dict:
    """
    GET JSON with retry on 429, 5xx, and network/DNS failures.
    """
    global session

    params = params or {}
    params.setdefault("format", "json")
    params.setdefault("api_key", API_KEY)

    last_status = None
    last_url = url
    last_exc = None

    for i in range(1, max_tries + 1):
        try:
            resp = session.get(url, params=params, timeout=60)
            last_status = resp.status_code
            last_url = resp.url

            if last_status == 200:
                _sleep_polite()
                return resp.json()

            # retryable HTTP
            if last_status == 429 or (500 <= last_status < 600):
                base_wait = min(BACKOFF_CAP_SEC, 2 ** (i - 1))
                wait = base_wait + random.random()
                print(f"HTTP {last_status} (try {i}/{max_tries}) sleeping {wait:.1f}s: {last_url}")
                time.sleep(wait)
                continue

            # non-retryable HTTP
            resp.raise_for_status()

        except (ConnectionError, Timeout, NameResolutionError) as e:
            # retryable network (DNS, connection reset, etc.)
            last_exc = e
            base_wait = min(BACKOFF_CAP_SEC, 2 ** (i - 1))
            wait = base_wait + random.random()
            print(f"NET ERROR (try {i}/{max_tries}) sleeping {wait:.1f}s: {url} ({e})")
            time.sleep(wait)

            # refresh session after a few failures
            if i in (3, 6, 9):
                try:
                    session.close()
                except Exception:
                    pass
                session = make_session()
            continue

    raise RuntimeError(f"Failed after {max_tries} tries. last_status={last_status} url={last_url} exc={last_exc}")


def paginate_list(first_url: str, first_params: dict, data_key: str) -> List[dict]:
    """
    Collect all items across pages by following pagination.next.
    """
    out: List[dict] = []
    url = first_url
    params = dict(first_params)

    while True:
        j = get_json(url, params=params)
        items = j.get(data_key, []) or []
        out.extend(items)

        nxt = (j.get("pagination") or {}).get("next")
        if not nxt:
            break

        url = nxt
        params = {"api_key": API_KEY, "format": "json"}

    return out


# =========================
# Bill parsing helpers
# =========================
BILL_URL_RE = re.compile(r"/v3/bill/(\d+)/([a-z]+)/(\d+)", re.IGNORECASE)
BILL_TEXT_RE = re.compile(
    r"\b(?P<prefix>H\.?R\.?|S\.?|H\.?J\.?RES\.?|S\.?J\.?RES\.?|H\.?CON\.?RES\.?|S\.?CON\.?RES\.?|H\.?RES\.?|S\.?RES\.?)\s*\.?\s*(?P<num>\d+)\b",
    re.IGNORECASE
)

def normalize_bill_type(prefix: str) -> Optional[str]:
    p = re.sub(r"[^A-Z]", "", prefix.upper())
    mapping = {
        "HR": "hr",
        "S": "s",
        "HJRES": "hjres",
        "SJRES": "sjres",
        "HCONRES": "hconres",
        "SCONRES": "sconres",
        "HRES": "hres",
        "SRES": "sres",
    }
    return mapping.get(p)

def parse_bill_from_url(url: str) -> Optional[Tuple[int, str, int]]:
    m = BILL_URL_RE.search(url or "")
    if not m:
        return None
    return int(m.group(1)), m.group(2).lower(), int(m.group(3))

def bill_display(bill_type: str, bill_number: int) -> str:
    t = bill_type.lower()
    if t == "hr": return f"H.R. {bill_number}"
    if t == "s": return f"S. {bill_number}"
    if t == "hjres": return f"H.J.Res. {bill_number}"
    if t == "sjres": return f"S.J.Res. {bill_number}"
    if t == "hconres": return f"H.Con.Res. {bill_number}"
    if t == "sconres": return f"S.Con.Res. {bill_number}"
    if t == "hres": return f"H.Res. {bill_number}"
    if t == "sres": return f"S.Res. {bill_number}"
    return f"{bill_type.upper()} {bill_number}"

def extract_bills_from_related_items(congress_hint: int, meeting_obj: dict) -> List[Tuple[int, str, int]]:
    """
    Extract (congress, billType, billNumber) from committeeMeeting.relatedItems.

    Congress.gov committee-meeting detail structure is:
      relatedItems -> bills -> bill -> { type, number, congress, url }
    (not a generic relatedItems.item[] list).
    """
    found: List[Tuple[int, str, int]] = []

    # --- Fallback: parse from the meeting title (often contains H.R. / S. citations)
    title = meeting_obj.get("title") or ""
    if isinstance(title, str):
        for m in BILL_TEXT_RE.finditer(title):
            btype = normalize_bill_type(m.group("prefix"))
            if btype:
                found.append((congress_hint, btype, int(m.group("num"))))

    related = meeting_obj.get("relatedItems") or meeting_obj.get("related_items") or {}
    if not isinstance(related, dict):
        return list(dict.fromkeys(found))

    # --- Primary: documented structure relatedItems.bills.bill[]
    bills_container = related.get("bills")
    bills_list = []

    if isinstance(bills_container, dict):
        bills_list = bills_container.get("bill") or bills_container.get("item") or bills_container.get("items") or []
    elif isinstance(bills_container, list):
        bills_list = bills_container

    if isinstance(bills_list, dict):
        bills_list = [bills_list]

    for b in bills_list:
        if not isinstance(b, dict):
            continue

        # Prefer parsing from URL if present (most authoritative)
        u = b.get("url") or b.get("URL")
        if isinstance(u, str) and u:
            parsed = parse_bill_from_url(u)
            if parsed:
                found.append(parsed)
                continue

        # Otherwise use explicit fields
        bcong = b.get("congress") or congress_hint
        btype_raw = b.get("type") or b.get("billType") or b.get("bill_type")
        bnum = b.get("number") or b.get("billNumber") or b.get("bill_number")

        try:
            bcong_int = int(bcong)
            bnum_int = int(str(bnum).strip())
        except Exception:
            continue

        btype = normalize_bill_type(str(btype_raw)) if btype_raw else None
        if btype:
            found.append((bcong_int, btype, bnum_int))

    # --- Backward compatibility: if API ever returns generic relatedItems.item[]
    # (keep your old logic as an extra fallback)
    items = related.get("item") or related.get("items") or []
    if isinstance(items, dict):
        items = [items]
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            u = it.get("url") or it.get("URL")
            if isinstance(u, str) and u:
                parsed = parse_bill_from_url(u)
                if parsed:
                    found.append(parsed)

            for fld in ["name", "title", "citation", "label", "text", "type"]:
                txt = it.get(fld)
                if isinstance(txt, str):
                    for m in BILL_TEXT_RE.finditer(txt):
                        btype = normalize_bill_type(m.group("prefix"))
                        if btype:
                            found.append((congress_hint, btype, int(m.group("num"))))

    # De-dupe, preserve order
    return list(dict.fromkeys(found))



# =========================
# Endpoint wrappers
# =========================
def fetch_committee_meeting_list(congress: int, chamber: str) -> List[dict]:
    url = f"{BASE}/committee-meeting/{congress}/{chamber}"
    params = {"limit": LIMIT, "offset": 0}
    return paginate_list(url, params, data_key="committeeMeetings")

def fetch_committee_meeting_item(congress: int, chamber: str, event_id: str) -> dict:
    url = f"{BASE}/committee-meeting/{congress}/{chamber}/{event_id}"
    return get_json(url)

def fetch_bill_detail(congress: int, bill_type: str, bill_number: int) -> dict:
    url = f"{BASE}/bill/{congress}/{bill_type}/{bill_number}"
    return get_json(url)


# =========================
# Checkpoint IO
# =========================
def month_csv_path(year: int, month: int) -> str:
    return os.path.join(OUT_DIR, f"hearings_{year}_{month:02d}.csv")

def month_done_path(year: int, month: int) -> str:
    return os.path.join(OUT_DIR, f"hearings_{year}_{month:02d}.done")

def month_is_done(year: int, month: int) -> bool:
    return os.path.exists(month_done_path(year, month))

def mark_month_done(year: int, month: int) -> None:
    with open(month_done_path(year, month), "w") as f:
        f.write(datetime.now(timezone.utc).isoformat(timespec="seconds"))

def load_month(year: int, month: int) -> Optional[pd.DataFrame]:
    p = month_csv_path(year, month)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            return df
        except Exception:
            return None
    return None

def save_month(df: pd.DataFrame, year: int, month: int) -> None:
    p = month_csv_path(year, month)
    df.to_csv(p, index=False)
    print(f"Saved: {p} (rows={len(df)})")

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Read a CSV safely:
      - skip header-only files
      - skip empty files
      - skip corrupt/truncated files
    """
    try:
        # header-only detection: <=1 non-empty line
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            nonempty = 0
            for _ in range(3):
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    nonempty += 1
            if nonempty <= 1:
                return None

        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"SKIP bad CSV: {path} ({e})")
        return None


# =========================
# MAIN RUNNER
# =========================
def run(years: Iterable[int]) -> pd.DataFrame:
    years_set: Set[int] = set(int(y) for y in years)
    target_bins = {(y, m) for y in years_set for m in range(1, 13)}
    done_bins = {(y, m) for (y, m) in target_bins if month_is_done(y, m)}
    print(f"Months requested: {len(target_bins)} | Already done (skip): {len(done_bins)}")

    # Cache bill (display, latest action)
    bill_cache: Dict[Tuple[int, str, int], Tuple[str, str]] = {}

    # Buffers per month
    buffers: Dict[Tuple[int, int], List[dict]] = {(y, m): [] for (y, m) in target_bins}

    for cong in range(CONGRESS_START, CONGRESS_END + 1):
        for chamber in MEETING_CHAMBERS:
            print(f"\n=== LIST committee meetings: congress={cong} chamber={chamber} ===")
            try:
                lst = fetch_committee_meeting_list(cong, chamber)
                print(f"Found {len(lst)} committee meeting list items for {cong}-{chamber}")
            except Exception as e:
                print(f"LIST FAILED {cong}-{chamber}: {e} (skipping)")
                continue

            for item in lst:
                event_id = item.get("eventId")
                if event_id is None:
                    continue

                # item detail (needed for type/date/relatedItems reliably)
                try:
                    cm = fetch_committee_meeting_item(cong, chamber, str(event_id))
                except Exception as e:
                    print(f"ITEM FAILED {cong}-{chamber}-{event_id}: {e}")
                    continue

                meeting = cm.get("committeeMeeting") or cm.get("committee_meeting") or {}
                if not meeting:
                    continue

                meeting_type = meeting.get("type") or ""
                meeting_type_l = meeting_type.strip().lower()

                # Congress.gov API: House types include Meeting/Hearing/Markup; Senate is always tagged "Meeting"
                # so "Meeting" for Senate can still represent hearings. :contentReference[oaicite:2]{index=2}
                if chamber == "house":
                    # keep only actual hearings (adjust if you also want markups/meetings)
                    if meeting_type_l not in {"hearing", "markup", "meeting"}:
                        continue
                elif chamber == "senate":
                    # keep senate "Meeting" items (these include hearings in senate data)
                    if meeting_type_l not in {"hearing", "markup", "meeting"}:
                        continue
                else:
                    # nochamber: be conservative
                    if meeting_type_l not in {"hearing", "markup", "meeting"}:
                        continue


                date = meeting.get("date") or meeting.get("meetingDate") or meeting.get("hearingDate")
                if not isinstance(date, str) or len(date) < 7:
                    continue

                try:
                    y = int(date[:4])
                    m = int(date[5:7])
                except Exception:
                    continue

                if y not in years_set:
                    continue
                if (y, m) in done_bins:
                    continue

                title = meeting.get("title") or ""
                committee = meeting.get("committee") or meeting.get("committeeName") or ""

                bills = extract_bills_from_related_items(cong, meeting)

                bill_displays: List[str] = []
                bill_latest: List[str] = []

                for (bcong, btype, bnum) in bills:
                    key = (bcong, btype, bnum)
                    if key not in bill_cache:
                        disp = bill_display(btype, bnum)
                        latest_str = ""
                        try:
                            bd = fetch_bill_detail(bcong, btype, bnum)
                            bill_obj = bd.get("bill", {}) or {}
                            latest_action = bill_obj.get("latestAction") or {}
                            ad = latest_action.get("actionDate") or ""
                            txt = latest_action.get("text") or ""
                            latest_str = (ad + " — " + txt).strip(" —")
                        except Exception:
                            pass
                        bill_cache[key] = (disp, latest_str)

                    disp, latest_str = bill_cache[key]
                    bill_displays.append(disp)
                    if latest_str:
                        bill_latest.append(latest_str)

                has_bill = 1 if bill_displays else 0

                buffers[(y, m)].append({
                    "eventId": int(event_id),
                    "congress": cong,
                    "api_chamber": chamber,
                    "meeting_type": meeting_type,
                    "date": date,
                    "year": y,
                    "month": m,
                    "title": title,
                    "committee": committee,
                    "has_bill": has_bill,
                    "bill_numbers": "; ".join(bill_displays) if bill_displays else "",
                    "bill_latest_actions": "; ".join(bill_latest) if bill_latest else "",
                    "update_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })

                # flush per month periodically
                if len(buffers[(y, m)]) >= 2000:
                    dfm = pd.DataFrame(buffers[(y, m)])
                    existing = load_month(y, m)
                    if existing is not None and not existing.empty:
                        dfm = pd.concat([existing, dfm], ignore_index=True).drop_duplicates(subset=["eventId"], keep="last")
                    save_month(dfm, y, m)
                    buffers[(y, m)] = []

    # final flush + done markers
    cols = [
        "eventId","congress","api_chamber","meeting_type","date","year","month","title","committee",
        "has_bill","bill_numbers","bill_latest_actions","update_timestamp_utc"
    ]

    for (y, m), rows in buffers.items():
        if (y, m) in done_bins:
            continue

        if rows:
            dfm = pd.DataFrame(rows)
            existing = load_month(y, m)
            if existing is not None and not existing.empty:
                dfm = pd.concat([existing, dfm], ignore_index=True).drop_duplicates(subset=["eventId"], keep="last")
        else:
            # write header-only CSV for consistency (safe combine will skip)
            dfm = pd.DataFrame(columns=cols)

        save_month(dfm, y, m)
        mark_month_done(y, m)

    # combine robustly
    files = [month_csv_path(y, m) for (y, m) in sorted(target_bins) if os.path.exists(month_csv_path(y, m))]

    dfs: List[pd.DataFrame] = []
    for f in files:
        d = safe_read_csv(f)
        if d is not None:
            dfs.append(d)

    master = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not master.empty:
        master = (
            master.drop_duplicates(subset=["eventId"])
                  .sort_values(["year", "month", "congress", "api_chamber"])
                  .reset_index(drop=True)
        )

    master.to_csv(FINAL_CSV, index=False)
    print(f"Saved MASTER: {FINAL_CSV} (rows={len(master)})")
    return master


if __name__ == "__main__":
    run(YEARS)
