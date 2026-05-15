# El Jurado de la Representación

A Kahoot-style web app for evaluating fairness and representation in AI-generated images. A facilitator sends text prompts to an image generation model; participants rate the results on six fairness criteria.

## Requirements

- Python 3.9+
- An [OpenAI](https://platform.openai.com) API key (for `gpt-image-2`) and/or a [fal.ai](https://fal.ai) API key (for Flux and other models)

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd representation_trial

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and fill in your keys:
#   OPENAI_API_KEY=...
#   FAL_API_KEY=...
```

## Running

```bash
source venv/bin/activate        # if not already active
python app.py
```

The app starts on **http://0.0.0.0:5000**.

| URL | Who uses it |
|---|---|
| `http://<host>:5000/` | Participants |
| `http://<host>:5000/admin` | Facilitator |

## Usage

1. Open `/admin`, select a model, enter one or more image prompts, click **Generate images**.
2. Participants open `/` — they see a spinner with a live counter while images are generating, then the gallery appears automatically.
3. Participants fill in the rating form and click **Enviar valoración**.
4. The facilitator can download session logs from the two buttons in the admin header.
5. Click **Reset session** to start a new round.

## Logs

Two log files are written to `logs/` after each session:

- `logs/admin.jsonl` — one entry per generation: datetime, question ID, model, prompts, per-image OK/ERR status.
- `logs/public.jsonl` — one entry per vote: datetime, question ID, client session ID, scores.

Both files share the `question_id` field so records can be joined.

## Adding models

Add an entry to `MODEL_PRESETS` in `app.py`:

```python
{
    "id":          "my-model",
    "label":       "Provider — Model name",
    "api_format":  "fal",           # "fal" or "openai"
    "base_url":    "https://fal.run/owner/model",
    "image_size":  "square_hd",     # fal.ai size string
    "api_key_env": "FAL_API_KEY",
    "auth_scheme": "Key",           # "Key" for fal.ai, "Bearer" for OpenAI
},
```
