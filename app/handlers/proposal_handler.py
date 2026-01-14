"""Proposal generation handler - main business logic"""

from typing import Optional

from app.config.settings import Settings

# Import traceable for LangSmith tracing if enabled
if Settings.LANGSMITH_ENABLED and Settings.LANGCHAIN_API_KEY:
    from langsmith import traceable
else:
    # Dummy decorator if LangSmith not enabled
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from app.services.ai_service import AIService
from app.services.pptx_service import PPTXService
from app.utils.file_utils import read_input_file
from app.utils.parser import parse_slides_content


class ProposalHandler:
    """Handler for proposal generation workflow"""
    
    def __init__(self, prompt_style: Optional[str] = None):
        """
        Initialize handler with required services
        
        Args:
            prompt_style: Prompt style to use (default: from settings)
        """
        prompt_style = prompt_style or Settings.PROMPT_STYLE
        self.ai_service = AIService(prompt_style=prompt_style)
        self.pptx_service = PPTXService()
    
    @traceable(name="generate_proposal", run_type="chain")
    def generate_proposal(
        self, 
        input_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate proposal from input file
        
        Args:
            input_file: Path to input file (default: from settings)
            
        Returns:
            Path to generated PowerPoint file, or None if error
        """
        input_file = input_file or Settings.INPUT_FILE
        
        # Step 1: Read input
        requirement = read_input_file(input_file)
        if not requirement:
            return None
        
        print(f"📄 Read requirements ({len(requirement)} characters)")
        print()
        
        # Step 2: Generate content with AI
        try:
            ai_content = self.ai_service.generate_proposal_content(requirement)
            print("✅ Received response from AI")
            print()
        except Exception as e:
            print(f"❌ Error calling AI: {e}")
            # Show appropriate error message based on provider
            if Settings.AI_PROVIDER == "azure":
                print("💡 Please check GITHUB_TOKEN (AZURE_TOKEN) in .env file")
            else:
                print("💡 Please check OPENAI_API_KEY in .env file")
            return None
        
        # Step 3: Parse slides
        slides = parse_slides_content(
            ai_content, 
            slide_titles=Settings.SLIDE_TITLES
        )
        print(f"📊 Parsed into {len(slides)} slides")
        print()
        
        # Step 4: Create PowerPoint
        try:
            output_file = self.pptx_service.create_presentation(slides)
            return output_file
        except Exception as e:
            print(f"❌ Error creating PowerPoint: {e}")
            return None

