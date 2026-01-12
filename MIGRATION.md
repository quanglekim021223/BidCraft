# Migration Guide

## From Old Structure to New Structure

The project has been refactored into a proper modular structure. Here's what changed:

### Old Structure
```
bidcraft/
├── generate_proposal.py  # Single monolithic file
├── input.txt
└── requirements.txt
```

### New Structure
```
bidcraft/
├── app/
│   ├── main.py              # Entry point (replaces generate_proposal.py)
│   ├── config/              # Configuration
│   ├── services/            # Business logic
│   ├── handlers/            # Request handlers
│   ├── utils/               # Utilities
│   └── models/              # Data models
├── tests/
└── ...
```

### How to Run

**Old way (deprecated):**
```bash
python generate_proposal.py
```

**New way:**
```bash
python -m app.main
# or
python app/main.py
```

**After installing as package:**
```bash
pip install -e .
bidcraft
```

### What Changed

1. **Configuration**: Moved to `app/config/settings.py` - all environment variables and settings in one place
2. **AI Service**: Separated into `app/services/ai_service.py`
3. **PowerPoint Service**: Separated into `app/services/pptx_service.py`
4. **File Utils**: Moved to `app/utils/file_utils.py`
5. **Parsing Logic**: Moved to `app/utils/parser.py`
6. **Main Handler**: Business logic in `app/handlers/proposal_handler.py`
7. **Entry Point**: `app/main.py` - clean entry point

### Benefits

- ✅ Better code organization
- ✅ Easier to test
- ✅ Easier to extend (add new services, handlers, etc.)
- ✅ Follows Python best practices
- ✅ Ready for future features (API, web interface, etc.)

