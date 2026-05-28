"""
TrackHR Call Analyzer — Hexbis Innovations
Uses: Groq Whisper-Large-v3 (transcription) + Groq LLaMA-3.3-70b (diarization, summary, flags)

Usage:
    python trackhr_analyzer.py
    python trackhr_analyzer.py /path/to/file.mp3
    python trackhr_analyzer.py --batch /folder
    python trackhr_analyzer.py --output ./my_reports
"""

import sys, json, time, argparse, tempfile
from pathlib import Path
from datetime import datetime

GROQ_API_KEY = "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"


WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL     = "llama-3.3-70b-versatile"
MAX_CHUNK_MB  = 24
CHUNK_MS      = 10 * 60 * 1000   

import groq as groq_sdk
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

def transcribe_audio(client, audio: AudioSegment, tmp_dir: Path) -> dict:
    print("\n[1/3] Transcribing...")
    all_segments, full_texts = [], []
    for idx, (chunk, offset_ms) in enumerate(split_if_needed(audio, tmp_dir / "chunk.mp3")):
        chunk_path = tmp_dir / f"chunk_{idx}.mp3"
        export_chunk(chunk, chunk_path)
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


def call_llm(client, prompt: str, retries=3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"LLM error ({e}), retrying in {wait}s...")
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

def diarize(client, segments: list) -> str:
    print("[2/3] Diarizing...")


    CHUNK_SIZE = 30
    chunks = [segments[i:i+CHUNK_SIZE] for i in range(0, len(segments), CHUNK_SIZE)]
    print(f"  Diarizing in {len(chunks)} chunk(s) of up to {CHUNK_SIZE} segments...")

    all_lines = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...")
        result = call_llm(client, DIARIZE_PROMPT.format(
            segments_json=json.dumps(chunk, indent=2, ensure_ascii=False)
        ))
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

def summarize(client, diarized_text: str) -> str:
    print("[3/3] Summarizing and extracting flags...")

    truncated = diarized_text[:8000]
    if len(diarized_text) > 8000:
        truncated += "\n\n[Transcript truncated for summary]"
    return call_llm(client, SUMMARY_PROMPT.format(diarized_text=truncated))


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

def extract_flags(client, diarized_text: str) -> str:
    truncated = diarized_text[:8000]
    if len(diarized_text) > 8000:
        truncated += "\n\n[Transcript truncated for analysis]"
    return call_llm(client, FLAGS_PROMPT.format(diarized_text=truncated))



def analyze(audio_path: Path, client, output_dir: Path):
    print(f"\nAnalyzing: {audio_path.name}")
    print("-" * 50)

    with tempfile.TemporaryDirectory(prefix="trackhr_") as tmp:
        tmp_dir = Path(tmp)
        audio         = load_audio(audio_path)
        transcription = transcribe_audio(client, audio, tmp_dir)
        diarized      = diarize(client, transcription["segments"])
        summary       = summarize(client, diarized)
        flags         = extract_flags(client, diarized)

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


def main():
    parser = argparse.ArgumentParser(description="TrackHR Call Analyzer — Hexbis Innovations")
    parser.add_argument("audio", nargs="?", help="Path to MP3 file")
    parser.add_argument("--batch",  metavar="FOLDER", help="Process all MP3s in a folder")
    parser.add_argument("--output", metavar="DIR", default="./trackhr_reports",
                        help="Output directory (default: ./trackhr_reports)")
    args = parser.parse_args()

    client     = groq_sdk.Groq(api_key=GROQ_API_KEY)
    output_dir = Path(args.output)

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
                analyze(mp3, client, output_dir)
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

    analyze(audio_path, client, output_dir)


if __name__ == "__main__":
    main()