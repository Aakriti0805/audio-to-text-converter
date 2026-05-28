# TrackHR Call Analyzer
### by Hexbis Innovations — Faridabad, Haryana

---

## Overview

TrackHR Call Analyzer is a terminal-based Python tool built for **Hexbis Innovations** to automatically process customer service call recordings related to their HR-management product, **TrackHR**. The tool takes raw MP3 audio files as input and produces a fully structured analysis report — no manual effort required.

---

## What It Does

The tool runs four steps automatically on every call recording:

**1. Transcription**
The audio is converted to text using **Groq's Whisper Large v3** model — one of the most accurate speech-to-text models available. Every word is transcribed along with its exact timestamp, split into segments throughout the call.

**2. Diarization**
The raw transcript is analyzed by **Groq's LLaMA 3.3 70B** language model, which identifies who is speaking at each moment — labeling every line as either `Agent` (the customer service representative) or `Customer`. The output is a clean, readable conversation with timestamps.

**3. Summary**
The same LLaMA model reads the diarized transcript and writes a concise 150–250 word summary of the call — covering the reason for the call, issues raised, solutions offered, and the final outcome.

**4. Feedback Flags**
The model then performs a structured CX (Customer Experience) analysis and extracts actionable flags across seven categories: customer sentiment, product issues, feature requests, agent performance, escalation status, priority level, and recommended next steps for the company.

---

## Tech Stack

| Component | Technology |
|---|---|
| Speech-to-Text | Groq — Whisper Large v3 |
| AI Analysis | Groq — LLaMA 3.3 70B Versatile |
|                  AND               |
| AI Analysis |Google AI Studio - Gemini 3.1 Flash-Lite |
| Audio Processing | pydub + ffmpeg |
| Language | Python 3.11+ |
| Interface | Terminal (CLI) |

---

## Key Features

- **Single file or batch processing** — analyze one call or an entire folder of MP3s in one command
- **Handles long recordings** — automatically splits audio files larger than 24 MB into 10-minute chunks before sending to the API
- **Token-safe diarization** — segments are processed in batches of 30 to stay within API rate limits, with automatic retries on failure
- **Auto-saved reports** — every analysis is saved as a timestamped `.txt` file in a dedicated reports folder
- **Single API key** — only Groq is needed; both Whisper and LLaMA run on the same platform

---

## Output

For every audio file, the tool produces:

1. **Diarized Transcript** — the full conversation with speaker labels and timestamps in `[MM:SS - MM:SS]  SPEAKER: text` format
2. **Call Summary** — a professional paragraph summarizing the entire call
3. **Feedback Flags** — a structured report with sentiment, issues, suggestions, agent performance notes, escalation recommendation, priority level, and action items for Hexbis

All three sections are printed in the terminal and saved together in a single report file.

---

## Use Case

This tool is designed for **Hexbis Innovations' quality assurance and customer success teams** to review large volumes of TrackHR support calls efficiently — without listening to every recording manually. It surfaces product problems, customer pain points, and agent performance issues automatically, enabling faster decisions and better product development.

---

## Usage

```bash
# Single file
python trackhr_analyzer.py /path/to/call.mp3

# Batch — process all MP3s in a folder
python trackhr_analyzer.py --batch /path/to/calls/

# Custom output folder
python trackhr_analyzer.py call.mp3 --output ./my_reports
```

---

*Built for Hexbis Innovations · TrackHR Customer Support Pipeline · 2026*
