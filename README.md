# 🩺 Health Assistant AI Chatbot

An intelligent, production-grade medical triage and health assistant chatbot. Built with **Streamlit**, **Pydantic AI**, **Groq Llama 3**, and real-time live local doctor discovery powered by **DuckDuckGo Search**.

---

## ✨ Features

- **🩺 Evidence-Based Clinical Triage**: Evaluates health symptoms with structured medical prudence, providing safe recovery steps, home remedies, and warning signs.
- **⚡ Two-Tier High Speed Architecture**:
  - *Tier 1 (Fast Intent Extraction)*: Uses `llama-3.1-8b-instant` to parse medical intent and identify specialist needs in milliseconds without consuming large rate limits.
  - *Tier 2 (Targeted Local Doctor Search)*: Searches DuckDuckGo for top clinics, hospitals, and specialists tailored to the user's specific city or locality.
  - *Tier 3 (Clinical Structuring)*: Uses `llama-3.3-70b-versatile` with Pydantic type validation for nuanced, reliable medical advice.
- **📍 Real-Time Location Discovery**: Multi-provider IP auto-detection with instant manual override to discover specialists nearby.
- **⚖️ Smart BMI Assessment**: Calculates Body Mass Index (BMI), visual status badges, and exact healthy target weight ranges tailored to height.
- **💧 Interactive Hydration Tracker**: Configurable daily water intake targets, quick logging (+250ml, +500ml), and progress tracking.
- **🍎 Daily Wellness Tips**: Daily actionable health advice generated and cached daily.
- **📥 Consultation Summary Export**: One-click download of the consultation notes to share with your personal healthcare provider.
- **🚨 Emergency Protocols**: Prominent emergency hotline badges (112 / 911) for urgent and life-threatening symptoms.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom glassmorphism styling and dark mode UI
- **AI Framework**: Pydantic AI & Groq Cloud SDK (`llama-3.3-70b-versatile` & `llama-3.1-8b-instant`)
- **Search Engine**: DDGS (DuckDuckGo Live Search)
- **Environment & Packaging**: Python 3.10+, uv, python-dotenv

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/REBDRA/Health-Assistant-Chatbot.git
cd Health-Assistant-Chatbot
```

### 2. Set Up Virtual Environment & Dependencies
Using `uv` (recommended):
```bash
uv venv
uv pip install -r requirements.txt
```

Or using standard `pip`:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Add your free Groq API key (get one from [Groq Console](https://console.groq.com/keys)):
```ini
GROQ_API_KEY=gsk_your_actual_api_key_here
```
*(Alternatively, you can enter your API key directly into the application's sidebar settings at runtime).*

### 4. Run the Application
```bash
python main.py
```
Or directly with Streamlit:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Architecture

```
├── .streamlit/
│   └── config.toml        # Theme and production server settings
├── ai_service.py          # Pydantic AI schemas, 2-tier classifier & DDGS doctor engine
├── app.py                 # Streamlit UI, wellness widgets, chat management
├── main.py                # CLI launcher
├── pyproject.toml         # Project metadata and package dependencies
├── requirements.txt       # Production dependencies
├── .env.example           # Environment template
└── README.md              # Documentation
```

---

## ⚠️ Medical Disclaimer

*Health Assistant AI is an informational triage and wellness assistant designed for educational and preliminary reference. It does NOT provide formal medical diagnoses or replace consultations with licensed physicians. In the event of a medical emergency, call **112** (India/EU), **911** (US), or visit the nearest emergency department immediately.*
