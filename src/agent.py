import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.recommender import Song, load_songs
except ModuleNotFoundError:
    from recommender import Song, load_songs

# Configure structured logging
logger = logging.getLogger("music_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class PlannedIntent:
    """Represents the parsed musical intent and constraints extracted from a user query."""
    raw_query: str
    target_genre: Optional[str] = None
    target_mood: Optional[str] = None
    target_energy: float = 0.5
    likes_acoustic: Optional[bool] = None
    min_tempo: Optional[float] = None
    max_tempo: Optional[float] = None
    target_valence: Optional[float] = None
    excluded_genres: List[str] = field(default_factory=list)
    excluded_artists: List[str] = field(default_factory=list)


@dataclass
class ScoredCandidate:
    """Represents a song evaluated against the planned intent with scoring details."""
    song: Song
    total_score: float
    confidence: float
    reasons: List[str]


@dataclass
class GuardrailResult:
    """Represents the outcome of guardrail verification checks."""
    passed: bool
    checks: Dict[str, bool]
    warnings: List[str]
    filtered_count: int


class IntentParser:
    """Parses natural language requests into structured music search parameters."""

    GENRE_KEYWORDS = {
        "pop": "pop",
        "indie pop": "indie pop",
        "lofi": "lofi",
        "lo-fi": "lofi",
        "study beats": "lofi",
        "rock": "rock",
        "metal": "metal",
        "ambient": "ambient",
        "jazz": "jazz",
        "synthwave": "synthwave",
        "classical": "classical",
        "hip-hop": "hip-hop",
        "hip hop": "hip-hop",
        "rap": "hip-hop",
        "r&b": "r&b",
        "rnb": "r&b",
        "folk": "folk",
        "electronic": "electronic",
        "techno": "electronic",
        "edm": "electronic",
        "blues": "blues",
        "reggae": "reggae",
    }

    MOOD_KEYWORDS = {
        "happy": ("happy", 0.8),
        "joyful": ("happy", 0.8),
        "chill": ("chill", 0.35),
        "relax": ("chill", 0.35),
        "relaxed": ("relaxed", 0.35),
        "calm": ("calm", 0.25),
        "focus": ("focused", 0.40),
        "focused": ("focused", 0.40),
        "studying": ("focused", 0.35),
        "study": ("focused", 0.35),
        "intense": ("intense", 0.90),
        "workout": ("intense", 0.90),
        "gym": ("intense", 0.92),
        "aggressive": ("aggressive", 0.95),
        "moody": ("moody", 0.70),
        "dark": ("moody", 0.75),
        "night drive": ("moody", 0.75),
        "confident": ("confident", 0.85),
        "soulful": ("soulful", 0.50),
        "reflective": ("reflective", 0.38),
        "energetic": ("energetic", 0.88),
        "upbeat": ("happy", 0.85),
        "melancholic": ("melancholic", 0.40),
        "sad": ("melancholic", 0.30),
        "uplifting": ("uplifting", 0.70),
    }

    @classmethod
    def parse(cls, query: str) -> PlannedIntent:
        """Extracts structured search intent from a natural language query."""
        q = query.lower()
        intent = PlannedIntent(raw_query=query)

        # Detect exclusions first
        exclusion_patterns = [
            (r"\b(?:no|without|exclude|except)\s+([a-z0-9\-]+)", 1),
        ]
        for pattern, group_idx in exclusion_patterns:
            for match in re.finditer(pattern, q):
                matched_word = match.group(group_idx).strip()
                if matched_word in cls.GENRE_KEYWORDS:
                    intent.excluded_genres.append(cls.GENRE_KEYWORDS[matched_word])

        # Detect genre (excluding any explicitly excluded genres)
        for keyword, genre in cls.GENRE_KEYWORDS.items():
            if genre in intent.excluded_genres:
                continue
            if re.search(r"\b(?<!no\s)(?<!without\s)(?<!exclude\s)" + re.escape(keyword) + r"\b", q):
                intent.target_genre = genre
                break

        # Detect mood and baseline energy
        for keyword, (mood, energy) in cls.MOOD_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", q):
                intent.target_mood = mood
                intent.target_energy = energy
                break

        # Detect acoustic preferences
        if "acoustic" in q or "unplugged" in q or "quiet" in q or "soft" in q:
            intent.likes_acoustic = True
            if intent.target_energy > 0.5:
                intent.target_energy = 0.4
        elif "electronic" in q or "synth" in q or "heavy" in q:
            intent.likes_acoustic = False

        # Detect energy modifiers
        if "high energy" in q or "very energetic" in q or "fast" in q or "pump up" in q:
            intent.target_energy = 0.90
        elif "low energy" in q or "sleep" in q or "slow" in q or "very calm" in q:
            intent.target_energy = 0.25

        return intent


class GuardrailEngine:
    """Verifies safety, catalog grounding, audio parameter bounds, and diversity."""

    PROMPT_INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"system override",
        r"delete database",
        r"drop table",
        r"<script>",
    ]

    @classmethod
    def validate_input(cls, query: str) -> Tuple[bool, Optional[str]]:
        """Checks for input sanity, length, and potential prompt injection."""
        if not query or not query.strip():
            return False, "Input query is empty."
        if len(query) > 500:
            return False, "Input query exceeds maximum length of 500 characters."
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Potentially unsafe input detected by guardrail: '{pattern}'."
        return True, None

    @classmethod
    def verify(
        cls,
        candidates: List[ScoredCandidate],
        catalog: List[Song],
        intent: PlannedIntent,
        max_artist_ratio: float = 0.5,
    ) -> GuardrailResult:
        """Performs multi-step verification on recommended candidates."""
        valid_song_ids = {s.id for s in catalog}
        checks = {
            "grounding": True,
            "energy_bounds": True,
            "exclusion_compliance": True,
            "diversity": True,
        }
        warnings = []

        # 1. Grounding check: verify no hallucinated songs
        for item in candidates:
            if item.song.id not in valid_song_ids:
                checks["grounding"] = False
                warnings.append(f"Grounding failure: Song ID {item.song.id} does not exist in catalog.")

        # 2. Audio bounds check
        for item in candidates:
            if not (0.0 <= item.song.energy <= 1.0):
                checks["energy_bounds"] = False
                warnings.append(f"Bounds failure: Song {item.song.title} has invalid energy {item.song.energy}.")

        # 3. Exclusion compliance
        for item in candidates:
            if item.song.genre in intent.excluded_genres:
                checks["exclusion_compliance"] = False
                warnings.append(f"Exclusion failure: Song {item.song.title} matches excluded genre {item.song.genre}.")

        # 4. Diversity check: prevent single artist domination
        if len(candidates) > 2:
            artist_counts: Dict[str, int] = {}
            for item in candidates:
                artist_counts[item.song.artist] = artist_counts.get(item.song.artist, 0) + 1
            max_allowed = max(1, int(len(candidates) * max_artist_ratio))
            for artist, count in artist_counts.items():
                if count > max_allowed:
                    checks["diversity"] = False
                    warnings.append(f"Diversity warning: Artist '{artist}' has {count} tracks (limit: {max_allowed}).")

        passed = all(checks.values())
        return GuardrailResult(
            passed=passed,
            checks=checks,
            warnings=warnings,
            filtered_count=len(candidates),
        )


class MusicCuratorAgent:
    """Agentic music workflow coordinating intent planning, catalog RAG retrieval, guardrails, and explanations."""

    def __init__(self, catalog_path: str = "data/songs.csv"):
        self.catalog_path = catalog_path
        self.catalog: List[Song] = self._load_catalog(catalog_path)
        logger.info("MusicCuratorAgent initialized with %d catalog tracks.", len(self.catalog))

    def _load_catalog(self, path: str) -> List[Song]:
        file_path = Path(path)
        if not file_path.is_absolute() and not file_path.exists():
            file_path = PROJECT_ROOT / path
        dict_songs = load_songs(str(file_path))
        return [
            Song(
                id=d["id"],
                title=d["title"],
                artist=d["artist"],
                genre=d["genre"],
                mood=d["mood"],
                energy=d["energy"],
                tempo_bpm=d["tempo_bpm"],
                valence=d["valence"],
                danceability=d["danceability"],
                acousticness=d["acousticness"],
            )
            for d in dict_songs
        ]

    def plan(self, query: str) -> PlannedIntent:
        """Step 1: Analyzes user query and formulates a search plan."""
        logger.info("Step 1 (Plan): Parsing intent for query: '%s'", query)
        intent = IntentParser.parse(query)
        logger.info(
            "Parsed Intent -> genre: %s, mood: %s, target_energy: %.2f, acoustic: %s",
            intent.target_genre,
            intent.target_mood,
            intent.target_energy,
            intent.likes_acoustic,
        )
        return intent

    def retrieve(self, intent: PlannedIntent, k: int = 5) -> List[ScoredCandidate]:
        """Step 2: RAG Retrieval - scores catalog songs against the planned intent."""
        logger.info("Step 2 (Retrieve/Act): Scoring catalog candidates against intent.")
        candidates: List[ScoredCandidate] = []

        for song in self.catalog:
            # Check exclusions first
            if song.genre in intent.excluded_genres or song.artist in intent.excluded_artists:
                continue

            score = 0.0
            reasons = []

            # Genre match (2.0 points)
            if intent.target_genre and song.genre.lower() == intent.target_genre.lower():
                score += 2.0
                reasons.append(f"genre match '{song.genre}' (+2.00)")
            elif intent.target_genre is None:
                # Neutral bonus when no genre is restricted
                score += 0.5
                reasons.append("open genre match (+0.50)")

            # Mood match (1.0 point)
            if intent.target_mood and song.mood.lower() == intent.target_mood.lower():
                score += 1.0
                reasons.append(f"mood match '{song.mood}' (+1.00)")

            # Energy similarity (up to 1.0 point)
            energy_diff = abs(song.energy - intent.target_energy)
            energy_sim = max(0.0, 1.0 - energy_diff)
            score += energy_sim
            reasons.append(f"energy similarity (+{energy_sim:.2f})")

            # Acoustic preference bonus (up to 0.5 point)
            if intent.likes_acoustic is not None:
                if intent.likes_acoustic and song.acousticness >= 0.6:
                    score += 0.5
                    reasons.append(f"high acousticness ({song.acousticness:.2f}) (+0.50)")
                elif not intent.likes_acoustic and song.acousticness < 0.4:
                    score += 0.5
                    reasons.append(f"low acousticness ({song.acousticness:.2f}) (+0.50)")

            # Valence alignment if target mood is happy/calm/sad
            if intent.target_mood in ["happy", "uplifting"] and song.valence >= 0.7:
                score += 0.3
                reasons.append(f"high valence ({song.valence:.2f}) (+0.30)")
            elif intent.target_mood in ["melancholic", "dark"] and song.valence <= 0.4:
                score += 0.3
                reasons.append(f"low valence ({song.valence:.2f}) (+0.30)")

            # Calculate confidence score between 0.0 and 1.0 (normalized against max possible score 4.8)
            confidence = min(1.0, score / 4.0)

            candidates.append(
                ScoredCandidate(
                    song=song,
                    total_score=round(score, 2),
                    confidence=round(confidence, 2),
                    reasons=reasons,
                )
            )

        # Sort descending by score
        ranked = sorted(candidates, key=lambda c: c.total_score, reverse=True)
        return ranked[:k]

    def verify(
        self, candidates: List[ScoredCandidate], intent: PlannedIntent
    ) -> Tuple[List[ScoredCandidate], GuardrailResult]:
        """Step 3: Verifies recommendations with guardrails and enforces diversity if needed."""
        logger.info("Step 3 (Verify): Running guardrails and checking safety/diversity.")
        guardrail_result = GuardrailEngine.verify(candidates, self.catalog, intent)

        # Enforce diversity if check failed: de-duplicate repeated artists
        final_candidates = candidates
        if not guardrail_result.checks["diversity"]:
            logger.warning("Diversity guardrail triggered. Re-ranking to ensure artist variety.")
            seen_artists: Dict[str, int] = {}
            adjusted = []
            deferred = []
            for item in candidates:
                count = seen_artists.get(item.song.artist, 0)
                if count < 1:
                    seen_artists[item.song.artist] = count + 1
                    adjusted.append(item)
                else:
                    deferred.append(item)
            final_candidates = (adjusted + deferred)[: len(candidates)]
            # Re-run guardrail check on adjusted set
            guardrail_result = GuardrailEngine.verify(final_candidates, self.catalog, intent)

        return final_candidates, guardrail_result

    def explain(self, candidates: List[ScoredCandidate], intent: PlannedIntent) -> str:
        """Step 4: Generates a clear explanation of why the playlist matches the request."""
        if not candidates:
            return "No songs in the catalog matched the specified criteria."

        top = candidates[0]
        avg_confidence = sum(c.confidence for c in candidates) / len(candidates)

        explanation = (
            f"Retrieved {len(candidates)} songs matching your request '{intent.raw_query}'. "
            f"The primary recommendation is '{top.song.title}' by {top.song.artist} because it aligns with "
            f"your preferences ({', '.join(top.reasons)}). "
            f"Overall recommendation confidence is {avg_confidence:.0%} across all retrieved tracks."
        )
        return explanation

    def run(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Executes the full agent workflow: Validate -> Plan -> Retrieve -> Verify -> Explain."""
        is_valid, error_msg = GuardrailEngine.validate_input(query)
        if not is_valid:
            logger.error("Input validation failed: %s", error_msg)
            return {
                "success": False,
                "error": error_msg,
                "intent": None,
                "recommendations": [],
                "guardrails": None,
                "explanation": f"Input rejected by guardrail: {error_msg}",
            }

        intent = self.plan(query)
        candidates = self.retrieve(intent, k=k)
        verified_candidates, guardrails = self.verify(candidates, intent)
        explanation = self.explain(verified_candidates, intent)

        return {
            "success": True,
            "error": None,
            "intent": intent,
            "recommendations": verified_candidates,
            "guardrails": guardrails,
            "explanation": explanation,
        }

    def apply_human_feedback(
        self, current_intent: PlannedIntent, feedback: str, k: int = 5
    ) -> Dict[str, Any]:
        """Allows human-in-the-loop refinement of current recommendations."""
        logger.info("Human-in-the-loop feedback received: '%s'", feedback)
        fb = feedback.lower()

        # Adapt existing intent based on feedback
        if "more energetic" in fb or "pump it up" in fb or "faster" in fb:
            current_intent.target_energy = min(1.0, current_intent.target_energy + 0.25)
        elif "calmer" in fb or "more chill" in fb or "slower" in fb:
            current_intent.target_energy = max(0.1, current_intent.target_energy - 0.25)

        if "more acoustic" in fb:
            current_intent.likes_acoustic = True
        elif "less acoustic" in fb or "electronic" in fb:
            current_intent.likes_acoustic = False

        if "no pop" in fb:
            current_intent.excluded_genres.append("pop")
        if "no rock" in fb:
            current_intent.excluded_genres.append("rock")
        if "no metal" in fb:
            current_intent.excluded_genres.append("metal")

        # Rerun retrieval with adapted intent
        candidates = self.retrieve(current_intent, k=k)
        verified_candidates, guardrails = self.verify(candidates, current_intent)
        explanation = self.explain(verified_candidates, current_intent)

        return {
            "success": True,
            "error": None,
            "intent": current_intent,
            "recommendations": verified_candidates,
            "guardrails": guardrails,
            "explanation": f"Refined after feedback ('{feedback}'): " + explanation,
        }
