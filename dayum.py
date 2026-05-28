"""
TrackHR Call Analyzer — Hexbis Innovations
Uses: Groq Whisper-Large-v3 (transcription) + Gemini 3.1 Flash Lite (diarization, summary, flags)

Usage:
    python trackhr_analyzer.py
    python trackhr_analyzer.py /path/to/file.mp3
    python trackhr_analyzer.py --batch /folder
    python trackhr_analyzer.py --output ./my_reports
"""

import sys, json, time, argparse, tempfile
from pathlib import Path
from datetime import datetime

GROQ_API_KEY   = "USE YOUR OWN API KEY PLS"
GEMINI_API_KEY = "USE YOUR OWN API KEY PLS"


WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL     = "gemini-3.1-flash-lite"
MAX_CHUNK_MB  = 24
CHUNK_MS      = 10 * 60 * 1000   


GROQ_WHISPER_USD_PER_HOUR = 0.111


GEMINI_INPUT_USD_PER_TOKEN  = 0.10  / 1_000_000
GEMINI_OUTPUT_USD_PER_TOKEN = 0.40  / 1_000_000


CHARS_PER_TOKEN = 4

USD_TO_INR = 96.17



class CostTracker:
    """Accumulates API usage and converts totals to INR."""

    def __init__(self, usd_to_inr: float = USD_TO_INR):
        self.usd_to_inr          = usd_to_inr
        self.audio_seconds       = 0.0   
        self.gemini_input_chars  = 0     
        self.gemini_output_chars = 0     
        self._gemini_calls       = 0

    
    def add_audio(self, duration_seconds: float):
        self.audio_seconds += duration_seconds

    def add_gemini_call(self, prompt: str, response: str):
        self._gemini_calls      += 1
        self.gemini_input_chars  += len(prompt)
        self.gemini_output_chars += len(response)

    
    @property
    def whisper_usd(self) -> float:
        return (self.audio_seconds / 3600) * GROQ_WHISPER_USD_PER_HOUR

    @property
    def gemini_input_usd(self) -> float:
        tokens = self.gemini_input_chars / CHARS_PER_TOKEN
        return tokens * GEMINI_INPUT_USD_PER_TOKEN

    @property
    def gemini_output_usd(self) -> float:
        tokens = self.gemini_output_chars / CHARS_PER_TOKEN
        return tokens * GEMINI_OUTPUT_USD_PER_TOKEN

    @property
    def total_usd(self) -> float:
        return self.whisper_usd + self.gemini_input_usd + self.gemini_output_usd

    @property
    def total_inr(self) -> float:
        return self.total_usd * self.usd_to_inr


    def summary(self) -> str:
        rate = self.usd_to_inr
        mins, secs = divmod(int(self.audio_seconds), 60)
        lines = [
            "",
            "=" * 72,
            " COST BREAKDOWN  (prices as of May 2025)",
            "=" * 72,
            f"  Exchange rate used          : $1 USD = ₹{rate:.2f}",
            "",
            f"  ┌─ Groq Whisper Large v3 ({'transcription'})",
            f"  │   Audio processed         : {mins}m {secs}s  ({self.audio_seconds:.1f}s)",
            f"  │   Rate                    : ${GROQ_WHISPER_USD_PER_HOUR}/hr",
            f"  │   Cost                    : ${self.whisper_usd:.6f}  (₹{self.whisper_usd * rate:.4f})",
            "",
            f"  ├─ Gemini Flash-Lite  ({self._gemini_calls} call(s))",
            f"  │   Input  chars / ~tokens  : {self.gemini_input_chars:,} / ~{self.gemini_input_chars // CHARS_PER_TOKEN:,}",
            f"  │   Output chars / ~tokens  : {self.gemini_output_chars:,} / ~{self.gemini_output_chars // CHARS_PER_TOKEN:,}",
            f"  │   Input  rate             : ${GEMINI_INPUT_USD_PER_TOKEN * 1_000_000:.2f}/M tokens",
            f"  │   Output rate             : ${GEMINI_OUTPUT_USD_PER_TOKEN * 1_000_000:.2f}/M tokens",
            f"  │   Input  cost             : ${self.gemini_input_usd:.6f}  (₹{self.gemini_input_usd * rate:.4f})",
            f"  │   Output cost             : ${self.gemini_output_usd:.6f}  (₹{self.gemini_output_usd * rate:.4f})",
            f"  │   Gemini subtotal         : ${self.gemini_input_usd + self.gemini_output_usd:.6f}  (₹{(self.gemini_input_usd + self.gemini_output_usd) * rate:.4f})",
            "",
            f"  └─ TOTAL                    : ${self.total_usd:.6f}  ≈  ₹{self.total_inr:.4f}",
            "=" * 72,
            "  Note: Token counts are estimated (~4 chars/token). Actual billing",
            "  may differ slightly. Update USD_TO_INR constant for current rate.",
            "=" * 72,
        ]
        return "\n".join(lines)



def fetch_usd_inr() -> float:
    """Fetch live mid-market USD/INR from exchangerate-api (no key needed)."""
    try:
        import urllib.request, json as _json
        url = "https://open.er-api.com/v6/latest/USD"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = _json.loads(r.read())
        rate = data["rates"]["INR"]
        print(f"[i] Live USD/INR fetched: ₹{rate:.2f}")
        return rate
    except Exception as e:
        print(f"[!] Could not fetch live rate ({e}). Using default ₹{USD_TO_INR:.2f}")
        return USD_TO_INR


import groq as groq_sdk
from google import genai
from pydub import AudioSegment


def load_audio(path: Path) -> AudioSegment:
    audio = AudioSegment.from_file(str(path))
    duration_s = len(audio) / 1000
    mins, secs = divmod(int(duration_s), 60)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Loaded: {path.name}  ({mins}m {secs}s, {size_mb:.1f} MB)")
    return audio

def export_chunk(audio: AudioSegment, tmp_path: Path) -> Path:
    audio.export(str(tmp_path), format="mp3", bitrate="64k")
    return tmp_path

def split_if_needed(audio: AudioSegment, base_tmp: Path):
    test_path = base_tmp.parent / "__size_test__.mp3"
    export_chunk(audio, test_path)
    size_mb = test_path.stat().st_size / (1024 * 1024)
    test_path.unlink(missing_ok=True)
    if size_mb <= MAX_CHUNK_MB:
        yield audio, 0
        return
    total_ms = len(audio)
    n_chunks = -(-total_ms // CHUNK_MS)
    print(f"File is large; splitting into {n_chunks} chunks...")
    for i in range(n_chunks):
        start = i * CHUNK_MS
        yield audio[start:min(start + CHUNK_MS, total_ms)], start

def transcribe_audio(client, audio: AudioSegment, tmp_dir: Path, cost: CostTracker) -> dict:
    print("\n[1/3] Transcribing...")
    all_segments, full_texts = [], []
    for idx, (chunk, offset_ms) in enumerate(split_if_needed(audio, tmp_dir / "chunk.mp3")):
        chunk_path = tmp_dir / f"chunk_{idx}.mp3"
        export_chunk(chunk, chunk_path)

        
        chunk_duration_s = len(chunk) / 1000
        cost.add_audio(chunk_duration_s)

        with open(chunk_path, "rb") as f:
            response = client.audio.transcriptions.create(
                file=(chunk_path.name, f, "audio/mpeg"),
                model=WHISPER_MODEL,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language="en",
            )
        chunk_path.unlink(missing_ok=True)
        offset_s = offset_ms / 1000
        for seg in response.segments:
            s = seg if isinstance(seg, dict) else vars(seg)
            all_segments.append({
                "start": round(s["start"] + offset_s, 2),
                "end":   round(s["end"]   + offset_s, 2),
                "text":  s["text"].strip(),
            })
        full_texts.append(response.text.strip())
    print(f"Transcription done — {len(all_segments)} segments.")
    return {"full_text": " ".join(full_texts), "segments": all_segments}

def call_llm(gemini_client, prompt: str, cost: CostTracker, retries=3) -> str:
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )
            result = response.text.strip()
            cost.add_gemini_call(prompt, result)   
            return result
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"Gemini error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

DIARIZE_PROMPT = """\
You are an expert call-center transcript analyst for Hexbis Innovations.

Below is a verbatim transcript of a customer service call about TrackHR \
(an HR-management product by Hexbis Innovations, Faridabad, Haryana).

The transcript has timestamped segments in JSON. Your job:
1. Identify the two speakers: "Agent" (customer-service rep) and "Customer".
2. Assign each segment to the correct speaker based on context, tone, and vocabulary.
3. Output ONLY the diarized transcript in this exact format per line:
   [MM:SS - MM:SS]  SPEAKER: text

No explanations. No extra text. Only the diarized lines.

SEGMENTS (JSON):
{segments_json}
"""

def format_ts(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"

def diarize(client, segments: list, cost: CostTracker) -> str:
    print("[2/3] Diarizing...")

    CHUNK_SIZE = 30
    chunks = [segments[i:i+CHUNK_SIZE] for i in range(0, len(segments), CHUNK_SIZE)]
    print(f"  Diarizing in {len(chunks)} chunk(s) of up to {CHUNK_SIZE} segments...")

    all_lines = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...")
        result = call_llm(client, DIARIZE_PROMPT.format(
            segments_json=json.dumps(chunk, indent=2, ensure_ascii=False)
        ), cost)
        all_lines.append(result.strip())
        if i < len(chunks) - 1:
            time.sleep(1)

    print("Diarization done.")
    return "\n".join(all_lines)

SUMMARY_PROMPT = """\
You are a senior QA analyst at Hexbis Innovations (makers of TrackHR, \
an HR-management SaaS, Faridabad, Haryana).

Write a concise SUMMARY (150-250 words) of the customer support call below covering:
- Reason for the call
- Key issues or pain points
- Solutions or steps offered by the Agent
- Outcome / resolution status
- Any follow-up actions promised

Be factual and professional.

DIARIZED TRANSCRIPT:
{diarized_text}
"""

def summarize(client, diarized_text: str, cost: CostTracker) -> str:
    print("[3/3] Summarizing and extracting flags...")
    truncated = diarized_text[:8000]
    if len(diarized_text) > 8000:
        truncated += "\n\n[Transcript truncated for summary]"
    return call_llm(client, SUMMARY_PROMPT.format(diarized_text=truncated), cost)


FLAGS_PROMPT = """\
You are a CX intelligence engine for Hexbis Innovations (TrackHR).

Analyze the transcript and extract structured feedback flags.

Use these exact section headers:

## CUSTOMER SENTIMENT
[Positive / Negative / Neutral / Mixed — brief reason]

## PRODUCT ISSUES FLAGGED
[Bullet list of TrackHR bugs, limitations, or confusions. "None detected" if absent.]

## CUSTOMER REQUESTS / FEATURE SUGGESTIONS
[Bullet list of improvement requests. "None detected" if absent.]

## AGENT PERFORMANCE FLAGS
[Bullet list of Agent behaviour to praise or improve. "None detected" if absent.]

## ESCALATION REQUIRED?
[Yes / No — one-line justification]

## PRIORITY LEVEL
[Critical / High / Medium / Low — one-line justification]

## RECOMMENDED ACTIONS FOR HEXBIS
[2-4 concrete next steps]

DIARIZED TRANSCRIPT:
{diarized_text}
"""

def extract_flags(client, diarized_text: str, cost: CostTracker) -> str:
    truncated = diarized_text[:8000]
    if len(diarized_text) > 8000:
        truncated += "\n\n[Transcript truncated for analysis]"
    return call_llm(client, FLAGS_PROMPT.format(diarized_text=truncated), cost)

def save_report(output_dir: Path, audio_name: str, diarized: str, summary: str,
                flags: str, cost_summary: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_name).stem.replace(" ", "_")
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"{stem}_analysis_{ts}.txt"
    out_file.write_text(
        f"TrackHR Call Analysis Report\n"
        f"File : {audio_name}\n"
        f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*72}\n\n"
        f"{'='*72}\n SECTION 1 — DIARIZED TRANSCRIPT\n{'='*72}\n\n{diarized}\n\n"
        f"{'='*72}\n SECTION 2 — CALL SUMMARY\n{'='*72}\n\n{summary}\n\n"
        f"{'='*72}\n SECTION 3 — FEEDBACK FLAGS & RECOMMENDATIONS\n{'='*72}\n\n{flags}\n\n"
        f"{'='*72}\n SECTION 4 — COST BREAKDOWN\n{'='*72}\n\n{cost_summary}\n",
        encoding="utf-8",
    )
    return out_file

def analyze(audio_path: Path, groq_client, gemini_client, output_dir: Path,
            usd_to_inr: float):
    print(f"\nAnalyzing: {audio_path.name}")
    print("-" * 50)

    cost = CostTracker(usd_to_inr=usd_to_inr)

    with tempfile.TemporaryDirectory(prefix="trackhr_") as tmp:
        tmp_dir = Path(tmp)
        audio         = load_audio(audio_path)
        transcription = transcribe_audio(groq_client, audio, tmp_dir, cost)
        diarized      = diarize(gemini_client, transcription["segments"], cost)
        summary       = summarize(gemini_client, diarized, cost)
        flags         = extract_flags(gemini_client, diarized, cost)

    sep = "=" * 72

    print(f"\n{sep}")
    print(" SECTION 1 — DIARIZED TRANSCRIPT")
    print(sep)
    print(diarized)

    print(f"\n{sep}")
    print(" SECTION 2 — CALL SUMMARY")
    print(sep)
    print(summary)

    print(f"\n{sep}")
    print(" SECTION 3 — FEEDBACK FLAGS & RECOMMENDATIONS")
    print(sep)
    print(flags)

    cost_summary = cost.summary()
    print(f"\n{sep}")
    print(" SECTION 4 — COST BREAKDOWN")
    print(cost_summary)

    out_file = save_report(output_dir, audio_path.name, diarized, summary, flags, cost_summary)
    print(f"\nReport saved: {out_file}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="TrackHR Call Analyzer — Hexbis Innovations")
    parser.add_argument("audio", nargs="?", help="Path to MP3 file")
    parser.add_argument("--batch",     metavar="FOLDER", help="Process all MP3s in a folder")
    parser.add_argument("--output",    metavar="DIR", default="./trackhr_reports",
                        help="Output directory (default: ./trackhr_reports)")
    parser.add_argument("--live-rate", action="store_true",
                        help="Fetch live USD/INR exchange rate before running")
    args = parser.parse_args()

    groq_client   = groq_sdk.Groq(api_key=GROQ_API_KEY)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    output_dir    = Path(args.output)

    usd_to_inr = fetch_usd_inr() if args.live_rate else USD_TO_INR
    if not args.live_rate:
        print(f"[i] Using hardcoded USD/INR rate: ₹{usd_to_inr:.2f}  "
              f"(pass --live-rate to fetch current rate)")

    if args.batch:
        folder = Path(args.batch)
        if not folder.is_dir():
            print(f"[!] Folder not found: {folder}")
            sys.exit(1)
        mp3_files = sorted(folder.glob("*.mp3"))
        if not mp3_files:
            print(f"[!] No MP3 files found in {folder}")
            sys.exit(1)
        for mp3 in mp3_files:
            try:
                analyze(mp3, groq_client, gemini_client, output_dir, usd_to_inr)
            except Exception as e:
                print(f"[!] Error processing {mp3.name}: {e}")
        return

    if args.audio:
        audio_path = Path(args.audio.strip().strip('"').strip("'"))
    else:
        raw = input("Enter audio file path: ").strip().strip('"').strip("'")
        audio_path = Path(raw)

    if not audio_path.exists():
        print(f"[!] File not found: {audio_path}")
        sys.exit(1)
    if audio_path.suffix.lower() != ".mp3":
        print(f"[!] Only MP3 files are supported. Got: {audio_path.suffix}")
        sys.exit(1)

    analyze(audio_path, groq_client, gemini_client, output_dir, usd_to_inr)


if __name__ == "__main__":
    main()
