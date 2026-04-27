# Task 3: General Health Query Chatbot (Prompt Engineering Based)

This task builds a safe, prompt-engineered chatbot for general health questions using an LLM.

## Objectives Covered

- Build a chatbot that sends user queries to an LLM
- Use prompt engineering for friendly and clear responses
- Add safety filtering for harmful/unsafe medical requests
- Support common query types like:
  - "What causes a sore throat?"
  - "Is paracetamol safe for children?"

## Tools Used

- Groq API (`llama3-8b-8192` default, configurable)
- OpenAI API (`gpt-3.5-turbo` default, configurable)
- Hugging Face Inference API (optional fallback)
- Python + requests + openai

## Project Structure

- `scripts/task3_health_query_chatbot.py` - chatbot script (interactive + self-test)
- `outputs/` - saved chat transcripts
- `requirements.txt` - dependencies

## Environment Variables

Create a `.env` file at `Task-3-HealthBot/.env` and set **only one provider at a time**.
You do **not** need both keys together.

### Groq (recommended)

```env
HEALTH_CHAT_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama3-8b-8192
```

### OpenAI

```env
HEALTH_CHAT_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
```

### Hugging Face

```env
HEALTH_CHAT_PROVIDER=huggingface
HUGGINGFACE_API_KEY=your_hf_api_key
HF_MODEL=gpt2
```

You can copy from `Task-3-HealthBot/.env.example`.

## How to Run

From `AI-ML-Tasks` root:

### 1) Quick validation (no API call)

```powershell
python .\run_task3.py
```

### 2) Start interactive chatbot

```powershell
python .\run_task3.py --chat
```

Type `exit` to close chat.

## Safety Behavior

- Blocks unsafe requests (e.g., self-harm, overdose, exact medication dosage)
- Adds a safety note to each response
- Encourages consulting licensed healthcare professionals for personal advice

## Example Prompts

- `What causes a sore throat?`
- `Is paracetamol safe for children?`
- `How can I reduce fever at home?`
