# This folder is for LLM related POC

---

## 🚀 POC
- How to add memory module in chatbot

---

## ⚙️ Setup Instructions

### 1. Create `.env` file in project root directory
Add the following environment variables:

```env
# Get your API key from https://console.groq.com
GROQ_API_KEY=<groq-token>
LLM_MODEL=llama-3.1-8b-instant
MAX_TOKENS_HISTORY=4000
```

### 2. Create virtual environment & install dependencies
Create a Python virtual environment and install the required packages. Run the below from commands from **project root directory**

```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

pip install -r poc/requirements.txt
```

### 3. Execute the script to initiate the chatbot

```bash
python .\poc\chatbot_with_memory\chatbot.py
```

