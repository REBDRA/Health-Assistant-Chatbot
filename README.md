<div align="center">

# 🩺 Health Assistant AI

### Intelligent Medical Triage & Verified Healthcare Discovery

An ultra-responsive, production-grade health assistant chatbot powered by **Groq LPU**, **Pydantic AI**, and real-time local doctor discovery.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-2.0+-E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white)](https://ai.pydantic.dev/)
[![Groq LPU](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036.svg?style=for-the-badge&logo=speedtest&logoColor=white)](https://groq.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-9%20Passing-success.svg?style=for-the-badge&logo=pytest&logoColor=white)](tests/test_health_app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

[**Explore Features**](#-key-features) • [**How to Access**](#-how-to-access--use) • [**Quick Start**](#-quick-start) • [**Architecture**](#-system-architecture) • [**Configuration**](#-configuration) • [**Testing**](#-testing)

---

</div>

## 🌟 Overview

**Health Assistant AI** bridges the gap between symptom confusion and actionable healthcare guidance. By pairing sub-second Groq inference with structured Pydantic validation and real-time live local doctor discovery, it delivers structured, evidence-based triage, home remedies, warning signs, and verified local doctor contact information in under 2 seconds.

Designed with a sleek **glassmorphic dark-mode interface**, the platform also bundles an all-in-one personal wellness suite featuring real-time BMI assessment, hydration tracking, daily health tips, and one-click consultation notes export.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🩺 Clinical Symptom Triage** | Evaluates health concerns using evidence-based medical schemas. Delivers clear explanations, safe home remedies, red-flag warning signs, and specialist recommendations. |
| **⚡ Multi-Tier AI Architecture** | Dual-model orchestration: lightweight intent extraction (`llama-3.1-8b-instant`) paired with high-precision clinical structuring (`llama-3.3-70b-versatile`). |
| **📍 Verified Doctor Discovery** | Real-time DuckDuckGo search combined with a pan-India verified hospital & clinic directory. Returns genuine contact numbers, ratings, addresses, and verified booking portals. |
| **🌍 Automatic Location Detection** | Multi-provider IP geo-location with seamless manual override for accurate local doctor recommendations anywhere in the world. |
| **⚖️ Smart BMI Assessment** | Interactive BMI calculator with color-coded classification badges and exact target healthy weight ranges computed for height. |
| **💧 Hydration Tracker** | Daily water intake tracker with customizable targets, quick one-click logging (+250ml, +500ml), and progress visualization. |
| **🍎 Daily Wellness Tips** | Science-backed daily health, nutrition, and exercise recommendations generated dynamically and cached per calendar day. |
| **📥 Consultation Summary Export** | Download your complete consultation summary as a formatted text file to share with your personal healthcare provider. |
| **🚨 Emergency Protocol Badges** | Instant one-tap access to emergency hotlines (**112** for India/EU, **911** for US, **108** for Ambulance) when critical symptoms are flagged. |

---

## 🌐 How to Access & Use

### 1. Launch the Application Locally
Once started (see [Quick Start](#-quick-start)), open your web browser and navigate to:

```text
http://localhost:8501
```

> 💡 **Network Access**: To access from a mobile device or other computer on the same local network, use the `Network URL` displayed in your terminal (e.g., `http://192.168.x.x:8501`).

### 2. Enter Your API Key (Two Options)
- **Option A (Recommended)**: Set `GROQ_API_KEY` in your `.env` file for automatic persistent login.
- **Option B (In-App)**: Open the left sidebar in the web interface, expand **Settings**, and paste your Groq API Key directly.

### 3. Set Your Location
- Health Assistant auto-detects your city via IP.
- To change or refine your location, enter your city or neighborhood in the sidebar's **Location** input (e.g., `Bandra, Mumbai` or `Manhattan, New York`). All nearby doctor recommendations will immediately adjust.

### 4. Chat & Wellness Tools
- **Consultation**: Type your symptoms or health queries into the chat input.
- **Wellness Tools**: Expand the **Health Suite** in the sidebar to calculate BMI or log hydration.
- **Download Notes**: Click the **Download Consultation Summary** button in the sidebar anytime to save your chat summary.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** installed
- A free **Groq API Key** (Get one in 30 seconds at [console.groq.com/keys](https://console.groq.com/keys))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/REBDRA/Health-Assistant-Chatbot.git
cd Health-Assistant-Chatbot

# 2. Create and activate a virtual environment
# On Windows:
python -m venv .venv
.venv\Scripts\activate

# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

*(Optional: If you use [uv](https://github.com/astral-sh/uv), simply run `uv venv` and `uv pip install -r requirements.txt` for blazing-fast setup).*

### Configuration

Copy the example environment template and add your API key:

```bash
# On Linux/macOS
cp .env.example .env

# On Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` with your preferred editor:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
DEFAULT_LOCATION=Kolkata, West Bengal, India
```

### Running the App

Run using the included runner:
```bash
python main.py
```

Or run directly with Streamlit:
```bash
streamlit run app.py
```

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([👤 User Prompt / Symptoms]) --> UI[🖥️ Streamlit Glassmorphic Frontend]
    UI --> T1[⚡ Tier 1: Fast Intent Classifier<br/><i>llama-3.1-8b-instant</i>]
    
    T1 -->|Extracts Specialty, Search Terms & Intent| SearchEngine{Needs Local Doctor?}
    
    SearchEngine -->|Yes| DDGS[🔎 DuckDuckGo Live Search Engine]
    SearchEngine -->|Yes| VDir[🏥 Verified Hospital Directory<br/><i>Apollo, Fortis, Max, AIIMS...</i>]
    
    DDGS --> Sanitize[🛡️ Domain & URL Sanitizer]
    VDir --> Sanitize
    
    Sanitize --> T2[🩺 Tier 2: Clinical Structuring Engine<br/><i>llama-3.3-70b-versatile + Pydantic AI</i>]
    SearchEngine -->|No| T2
    
    T2 --> TriageResp[📋 Validated HealthResponse Schema]
    TriageResp --> UI
    
    subgraph Wellness Suite [🧘 In-App Wellness Suite]
        BMI[⚖️ Smart BMI Calculator]
        Hydration[💧 Hydration Tracker]
        Tips[🍎 Daily Wellness Tips]
        Export[📥 Consultation Summary Exporter]
    end
    Wellness Suite -.-> UI
```

---

## ⚙️ Configuration

Environment variables can be defined in a `.env` file at the root of the project:

| Variable | Type | Required | Description | Default |
| :--- | :---: | :---: | :--- | :--- |
| `GROQ_API_KEY` | String | **Yes\*** | Groq Cloud API Key for high-speed inference. *(Can also be entered directly in UI sidebar)* | `None` |
| `DEFAULT_LOCATION` | String | No | Fallback location string for doctor search if IP detection is offline. | `Kolkata, West Bengal, India` |

---

## 🧪 Testing

The repository contains an automated test suite verifying schema validation, doctor search sanitization, URL filtering, and directory completeness:

```bash
# Run tests with pytest
python -m pytest
```

All 9 unit and integration tests run with standard assertion output:
```text
tests/test_health_app.py .........                               [100%]
============================== 9 passed ==============================
```

---

## 📁 Repository Structure

```
Health-Assistant-Chatbot/
├── .devcontainer/         # Dev container configuration for VS Code & GitHub Codespaces
├── .streamlit/
│   └── config.toml        # Streamlit dark theme & production server settings
├── tests/
│   └── test_health_app.py # Pytest test suite (Schemas, sanitizers, doctor directory)
├── ai_service.py          # Dual-tier Pydantic AI engine, doctor search & verified directory
├── app.py                 # Streamlit UI, glassmorphism styles, chat engine, health suite
├── main.py                # Cross-platform application entry point
├── pyproject.toml         # Build system & package metadata
├── requirements.txt       # Production dependencies
├── .env.example           # Template for environment configuration
└── README.md              # Project documentation
```

---

## 🛡️ Data Privacy & Security

- **Zero Data Retention**: Chat history lives entirely in your browser session state (`st.session_state`) and is never sent to third-party databases.
- **Direct Groq API Communication**: Requests travel directly to Groq's high-speed inference endpoints using secure TLS.
- **Sanitized Doctor Search**: External queries strip personal medical information and search strictly for medical specialties and location names.

---

## ⚠️ Medical Disclaimer

> [!WARNING]
> **Health Assistant AI is designed for informational and educational triage purposes only.**
> It does **NOT** provide definitive medical diagnoses, prescriptions, or treatment plans, and must not replace professional clinical judgment from licensed healthcare providers.
>
> **If you are experiencing severe pain, chest tightness, shortness of breath, or any medical emergency, please call your local emergency services immediately:**
> - 🇮🇳 **India**: `112` or `108` (Ambulance)
> - 🇺🇸 **United States / Canada**: `911`
> - 🇪🇺 **Europe / UK**: `112` / `999`

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the triage schemas, expand the verified hospital directory, or add new wellness widgets:

1. Fork the Project (`gh repo fork REBDRA/Health-Assistant-Chatbot`)
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">

⭐ **If you found this project helpful, please consider starring the repository!** ⭐

Distributed under the MIT License. Built with ❤️ for accessible healthcare assistance.

</div>
