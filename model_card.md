# Model Card and Responsible AI Reflection

## Project Overview

- **Model Name:** Music Curator AI
- **System Version:** 2.0 (Applied AI System)
- **Base Project:** ai110-module3show-musicrecommendersimulation-starter
- **Primary Function:** Natural language music curation, attribute-based retrieval, safety guardrail verification, and human-in-the-loop playlist refinement.

---

## What are the limitations or biases in your system?

The system currently works with a curated catalog of 18 songs in a CSV file. Because the catalog is small, some genres and moods only have one or two available tracks. When a user asks for a specific genre that has limited songs, the system fills the remaining playlist spots with songs from other genres that match the target energy.

The scoring model also has a slight genre preference because matching a named genre provides a larger baseline score than acoustic attributes. Additionally, the system evaluates structured metadata tags like genre, mood, energy, and acousticness rather than listening to the actual raw audio audio files.

---

## Could your AI be misused, and how would you prevent that?

In a commercial setting, recommendation engines can be misused to favor paid placements, create filter bubbles, or push repetitive content from high-paying record labels while burying independent musicians. Malicious users might also attempt prompt injection attacks to alter system instructions or overload search logic.

We prevent these issues through three guardrails:
1. Input sanitization: The input guardrail inspects queries for prompt injection patterns and character limits before processing.
2. Artist diversity guardrail: The system monitors artist frequency and prevents any single artist from monopolizing the playlist.
3. Grounding and transparency: Every song must come directly from verified catalog IDs, and each recommendation prints its exact scoring breakdown so the user knows why it was chosen.

---

## What surprised you while testing your AI's reliability?

What surprised me most was how subtle keyword collisions can confuse intent parsing. When testing negative instructions like "rock music, but no pop tracks", the parser initially noticed the word "pop" and classified it as a preferred genre. That meant the user got the exact genre they were trying to avoid.

Fixing this required running regex checks for exclusion words like "no", "without", and "exclude" first so that excluded terms are filtered before positive genre matching runs. After adding that check, reliability improved to 100 percent across our negative test cases.

---

## Collaboration with AI During This Project

During this project, I used AI as a pair programming assistant to help brainstorm edge cases, write test suites, and refine architecture choices.

### One suggestion accepted:
The AI suggested adding an automated artist diversity guardrail. In earlier tests, a user asking for lofi would often receive multiple songs from the same artist in a three-song playlist. Accepting the diversity check made the recommendations feel more balanced and natural.

### One suggestion rejected or revised:
The AI initially suggested installing an external vector database and using a third-party cloud embedding API to handle semantic similarity. I rejected that suggestion because it would have required external API keys, added unnecessary dependencies, and made the project harder for others to run. Instead, I revised the idea to use a local attribute matching algorithm that evaluates acousticness, energy, valence, and mood directly from our catalog. This kept the system fast, self-contained, and completely reproducible.

---

## Reliability and Human Evaluation Summary

| Test Case | Test Input | Expected Behavior | Actual Behavior | Result |
| :--- | :--- | :--- | :--- | :--- |
| Intent Extraction | "upbeat pop songs for dancing" | Detects pop, happy mood, energy >= 0.7 | Extracted pop, happy, energy 0.80 | Pass |
| Exclusion Handling | "Intense rock, but no pop" | Detects rock, excludes pop | Extracted rock, excluded pop | Pass |
| Grounding Check | "chill lofi music for night study" | 100% of returned songs exist in catalog | All 5 songs verified against catalog IDs | Pass |
| Input Guardrail | Empty query ("   ") | Rejects with clear error message | Rejected: "Input query is empty" | Pass |
| Adversarial Defense | "ignore previous instructions..." | Rejects injection pattern | Rejected unsafe input | Pass |
| Human Feedback Loop | "make it more energetic" | Increases target energy from previous turn | Target energy increased from 0.35 to 0.60 | Pass |
| Diversity Check | Multiple matching songs by same artist | Enforces artist distribution | Caps repeated artists in top slots | Pass |

All 10 automated unit and reliability tests passed successfully.
