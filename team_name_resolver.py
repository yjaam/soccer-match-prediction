import re
import unicodedata
from typing import Dict, Iterable, Optional, Tuple
from rapidfuzz import process, fuzz

STOPWORDS = {
    "fc", "sc", "sv", "vfb", "vfl", "bv", "bvb", "tsg", "eintracht",
    "borussia", "verein", "fur", "fuer", "leibesubungen", "und", "sportverein",
    "fussball", "fuball", "club", "clubb", "ev", "ag", "gmbh", "von", "zu"
}

# Add/extend with your known difficult mappings:
ALIASES = {
    "fc bayern munchen": "bayern munich",
    "bayern munchen": "bayern munich",
    "bayern münchen": "bayern munich",
    "borussia monchengladbach": "b monchengladbach",
    "1 fc koln": "fc koln",
    "koln": "fc koln",
    "vfl bochum 1848": "bochum",
    "fsv mainz 05": "mainz 05",
    "1 fsv mainz 05": "mainz 05",
    "tsg 1899 hoffenheim": "hoffenheim",
    "bayer 04 leverkusen": "bayer leverkusen",
    "rasenballsport leipzig": "rb leipzig",
    "st pauli": "st pauli",
    "fc st pauli": "st pauli",
}

def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def normalize_team_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).lower().strip()
    s = strip_accents(s)
    s = s.replace("&", " and ")
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # normalize common numeric prefixes
    s = s.replace("1 fc ", "fc ")
    s = s.replace("1 fsv ", "fsv ")

    return s

def canonical_tokens(name: str):
    n = normalize_team_name(name)
    toks = [t for t in n.split() if t and t not in STOPWORDS]
    return toks

def canonical_key(name: str) -> str:
    n = normalize_team_name(name)
    if n in ALIASES:
        n = ALIASES[n]
    toks = canonical_tokens(n)
    return " ".join(toks) if toks else n

class TeamNameResolver:
    def __init__(self, target_names: Iterable[str], min_fuzzy_score: int = 82):
        self.target_names = list(target_names)
        self.min_fuzzy_score = min_fuzzy_score
        self.cache: Dict[str, Optional[str]] = {}

        # Build canonical index for targets
        self.target_canonical: Dict[str, str] = {}
        for t in self.target_names:
            self.target_canonical[t] = canonical_key(t)

        # Reverse index canonical -> list of targets
        self.by_canonical: Dict[str, list] = {}
        for t, c in self.target_canonical.items():
            self.by_canonical.setdefault(c, []).append(t)

    def _token_overlap_score(self, a: str, b: str) -> float:
        sa, sb = set(canonical_tokens(a)), set(canonical_tokens(b))
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union

    def resolve(self, source_name: str) -> Optional[str]:
        if source_name in self.cache:
            return self.cache[source_name]

        src_norm = normalize_team_name(source_name)
        src_can = canonical_key(source_name)

        # 1) exact canonical match
        if src_can in self.by_canonical:
            # if multiple, pick shortest (usually cleaner canonical)
            cand = sorted(self.by_canonical[src_can], key=len)[0]
            self.cache[source_name] = cand
            return cand

        # 2) best token-overlap
        best_name, best_overlap = None, 0.0
        for t in self.target_names:
            ov = self._token_overlap_score(source_name, t)
            if ov > best_overlap:
                best_overlap, best_name = ov, t

        # Accept strong overlap directly
        if best_overlap >= 0.60:
            self.cache[source_name] = best_name
            return best_name

        # 3) fuzzy on canonical strings
        choices = {t: self.target_canonical[t] for t in self.target_names}
        # rapidfuzz expects list; we map back by score
        fuzzy_match = process.extractOne(
            src_can,
            list(choices.values()),
            scorer=fuzz.WRatio
        )

        if fuzzy_match:
            matched_can, score, _ = fuzzy_match
            if score >= self.min_fuzzy_score:
                # find first target with this canonical
                for t, c in choices.items():
                    if c == matched_can:
                        self.cache[source_name] = t
                        return t

        # 4) fallback: normalized exact (very loose)
        for t in self.target_names:
            if normalize_team_name(t) == src_norm:
                self.cache[source_name] = t
                return t

        self.cache[source_name] = None
        return None