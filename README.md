# BidCraft

> "Craft winning bids, faster."

AI-powered tool to automatically generate Proposal (PPTX/DOCX) from client requirements.

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create `.env` file from template:

```bash
cp .env.example .env
```

Then open `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Prepare input

Open `input.txt` file and paste the client's project requirements there.

## 📝 Usage

Run the application:

```bash
python -m app.main
```

Or from the project root:

```bash
python app/main.py
```

The application will:
1. Read requirements from `input.txt`
2. Send to GPT-4 to generate content for 5 slides
3. Create PowerPoint file: `proposal_YYYYMMDD_HHMMSS.pptx`

## 📊 Output

The PowerPoint file will have 5 slides:
- **INTRODUCTION**: Company introduction and capabilities
- **PROBLEM STATEMENT**: Analysis of client's current challenges
- **SOLUTION**: Specific proposed solution
- **TECHNOLOGY STACK**: Technologies to be used
- **TIMELINE**: Project implementation roadmap

## 📁 Project Structure

```
bidcraft/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config/                 # Configuration
│   │   ├── __init__.py
│   │   └── settings.py        # App settings & env vars
│   ├── services/              # Business logic services
│   │   ├── __init__.py
│   │   ├── ai_service.py      # AI content generation
│   │   └── pptx_service.py    # PowerPoint creation
│   ├── handlers/              # Request handlers
│   │   ├── __init__.py
│   │   └── proposal_handler.py # Main proposal workflow
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── file_utils.py      # File I/O operations
│   │   └── parser.py          # Content parsing
│   └── models/                # Data models
│       └── __init__.py
├── tests/                     # Test files
├── input.txt                  # Input requirements
├── requirements.txt
├── .env
└── README.md
```

## 🛠 Tech Stack

- Python 3.8+
- LangChain + OpenAI API
- python-pptx
