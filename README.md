# Iterative Copywriter

A Docker-based [Langflow](https://langflow.org/) project implementing the iterative copy refinement framework from [LLM-driven Constrained Copy Generation through Iterative Refinement](https://arxiv.org/html/2504.10391v1).

The framework generates marketing copy under constraints (length, keywords, tone of voice) and iteratively refines copies that fail evaluation, significantly improving the success rate.

## Architecture

```
Input (topic, constraints, persona)
        │
        ▼
  ┌─────────────┐
  │  Generator   │  ← OpenAI batch generation
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Formatter   │  ← Rule-based cleanup
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Evaluator   │  ← Length + keywords + tone checks
  └──────┬──────┘
     ┌───┴───┐
     │       │
   Pass    Fail + feedback
     │       │
     ▼       ▼
  Accept  ┌─────────┐
          │ Refiner  │  ← OpenAI refinement
          └────┬────┘
               │
               └──→ back to Formatter (up to J attempts)
```

## Quick Start

```bash
cp .env.example .env
```

Edit `.env` to add your OpenAI API key, then:

```bash
docker compose up --build
```

Open Langflow at [http://localhost:7860](http://localhost:7860).

## Custom Components

Five custom Langflow components are available in the sidebar under the custom category:

| Component | Purpose |
|---|---|
| **Copy Generator** | Generates a batch of marketing copies via OpenAI |
| **Copy Formatter** | Applies rule-based formatting (brand cleanup, punctuation) |
| **Constraint Evaluator** | Checks copy against length, keyword, and tone constraints |
| **Copy Refiner** | Refines failed copy using evaluation feedback via OpenAI |
| **Iterative Refinement Pipeline** | Orchestrates the full generate-evaluate-refine loop |

### Using the Pipeline Component

The **Iterative Refinement Pipeline** is the main entry point. It runs the complete loop internally:

1. Drag it onto the canvas
2. Fill in the inputs:
   - **OpenAI API Key**: your key (or use the environment variable)
   - **Topic**: e.g. "free delivery from store"
   - **Persona**: e.g. "budget-conscious families"
   - **Tone**: e.g. "confident, friendly, concise"
   - **Copy Structure**: `header` or `header_subheader`
   - **Length Constraints**: JSON like `{"header": {"min": 30, "max": 60}}`
   - **Required/Excluded Keywords**: JSON arrays
   - **Batch Size**: number of copies to generate (default 5)
   - **Max Refinements**: iteration limit per copy (default 2)
3. Connect its output to a **Text Output** or **Chat Output** node
4. Run the flow

### Using Individual Components

You can also build custom flows by wiring the individual components together:

1. **Copy Generator** → generates raw copies
2. **Copy Formatter** → cleans up each copy
3. **Constraint Evaluator** → checks constraints
4. **Copy Refiner** → fixes failed copies

### Importing the Example Flow

An example flow configuration is available at `flows/copy_refinement.json`. To import:

1. Open Langflow at http://localhost:7860
2. Click "New Flow" or use the import button
3. Select `copy_refinement.json`
4. Fill in your OpenAI API key and adjust parameters

## Example Configuration

A Campaign-A style configuration (single header, free delivery):

```json
{
  "topic": "free delivery from store",
  "persona": "general audience",
  "tone": "confident, friendly, concise",
  "copy_structure": "header",
  "length": {"header": {"min": 30, "max": 60}},
  "keywords_required": ["free delivery"],
  "keywords_excluded": ["cheap", "discount"],
  "batch_size": 5,
  "max_refinements": 2
}
```

A Campaign-B style configuration (header + subheader):

```json
{
  "topic": "free shipping with no order minimum",
  "persona": "returning customers",
  "tone": "confident, friendly, value-driven",
  "copy_structure": "header_subheader",
  "length": {
    "header": {"min": 20, "max": 40},
    "subheader": {"min": 50, "max": 80}
  },
  "keywords_required": ["free shipping"],
  "keywords_excluded": ["cheap"],
  "batch_size": 10,
  "max_refinements": 2
}
```

## Project Structure

```
iterative-copywriter/
├── docker-compose.yml          # Langflow service
├── Dockerfile                  # Custom image with components
├── .env                        # OpenAI credentials
├── .env.example                # Credential template
├── components/
│   ├── copy_generator.py       # Batch copy generation
│   ├── copy_formatter.py       # Rule-based formatting
│   ├── constraint_evaluator.py # Constraint checking
│   ├── copy_refiner.py         # Feedback-driven refinement
│   └── iterative_pipeline.py   # Full pipeline orchestrator
└── flows/
    └── copy_refinement.json    # Example flow configuration
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model to use for generation/evaluation |

## Reference

Based on: Vasudevan et al., "LLM-driven Constrained Copy Generation through Iterative Refinement" ([arXiv:2504.10391](https://arxiv.org/html/2504.10391v1))
