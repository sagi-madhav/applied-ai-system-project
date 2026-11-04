# System Architecture Diagram

This document contains the Mermaid source code for the Applied AI Music Recommender architecture.

```mermaid
flowchart TD
    subgraph UserInterface ["User Layer"]
        User["User"]
        App["Application Interface (CLI / Streamlit App)"]
    end

    subgraph AgenticCore ["Agentic AI Workflow"]
        InputGuard["Input Guardrail (Sanitization & Safety Check)"]
        Planner["Intent Planner (Extracts Genre, Mood, Energy, Exclusions)"]
        RAGRetriever["RAG Catalog Retriever (Attribute Scoring & Similarity)"]
        OutputGuard["Output Guardrails (Grounding, Bounds, Diversity Checks)"]
        Explainer["Explanation Generator (Natural Language Reasoning)"]
    end

    subgraph DataStore ["Data Layer"]
        Catalog[("Song Catalog CSV Database")]
    end

    subgraph HumanAndEvaluation ["Verification and Feedback"]
        FeedbackLoop["Human-in-the-Loop Feedback (User Playlist Refinement)"]
        AutomatedTests["Automated Reliability Suite (Pytest: Grounding, Guardrails, Determinism)"]
    end

    %% Data flow
    User -->|"Submits natural language query"| App
    App -->|"Sends raw prompt"| InputGuard
    InputGuard -->|"Validated prompt"| Planner
    InputGuard -.->|"Rejection error if unsafe"| App

    Planner -->|"Structured search parameters"| RAGRetriever
    Catalog -->|"Song records and audio attributes"| RAGRetriever

    RAGRetriever -->|"Ranked candidate songs"| OutputGuard
    OutputGuard -->|"Verified songs & confidence scores"| Explainer
    OutputGuard -.->|"Re-ranks if diversity check flags duplicate artist"| OutputGuard

    Explainer -->|"Formatted recommendations with match explanations"| App
    App -->|"Delivers final playlist"| User

    %% Human in the loop flow
    User -->|"Provides refinement feedback"| FeedbackLoop
    FeedbackLoop -->|"Updates intent parameters"| Planner

    %% Testing flow
    AutomatedTests -->|"Tests system components"| AgenticCore
```

## System Workflow Description

1. **User Layer**: The user enters a natural language music preference into either the terminal CLI or the Streamlit web app.
2. **Input Guardrail**: The query is validated for length, completeness, and prompt safety. Unsafe or empty queries are blocked immediately.
3. **Intent Planner**: The agent analyzes the user prompt to identify target genre, mood, target energy, acoustic preference, and excluded genres.
4. **RAG Catalog Retrieval**: The retriever pulls song records from the catalog database and computes similarity scores using audio features (valence, tempo, energy, acousticness) and metadata tags.
5. **Output Guardrails**: The candidates are verified for 100% catalog grounding (no fabricated songs), valid energy ranges, and artist diversity.
6. **Explanation Generator**: The agent builds a natural language explanation describing why the songs were picked and reports overall recommendation confidence.
7. **Human-in-the-Loop Refinement**: If the user wants adjustments (such as higher energy or less acoustic sound), the agent updates the search parameters and re-curates the playlist.
8. **Automated Reliability Suite**: Pytest tests verify that all components operate accurately, deterministically, and safely.
