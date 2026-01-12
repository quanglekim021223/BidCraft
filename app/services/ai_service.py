"""AI service for content generation"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from app.config.settings import Settings


class AIService:
    """Service for AI-powered content generation"""
    
    def __init__(self):
        """Initialize AI service with configured settings"""
        Settings.validate()
        self.llm = ChatOpenAI(
            model=Settings.OPENAI_MODEL,
            temperature=Settings.OPENAI_TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY
        )
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Setup the prompt template for proposal generation"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional Solution Architect at a technology company.
Writing style: professional, concise, technical-focused, avoid clichés.
Please write content for 5 proposal slides based on the project requirements.

Output format: Each slide separated by "---SLIDE---"
Slide 1: INTRODUCTION - Company introduction and capabilities
Slide 2: PROBLEM STATEMENT - Analysis of client's current challenges
Slide 3: SOLUTION - Specific proposed solution
Slide 4: TECHNOLOGY STACK - Technologies to be used
Slide 5: TIMELINE - Project implementation roadmap

Each slide should have 3-5 bullet points, concise and easy to understand."""),
            ("user", "Project requirements:\n\n{requirement}")
        ])
    
    def generate_proposal_content(self, requirement_text: str) -> str:
        """
        Generate proposal content using AI
        
        Args:
            requirement_text: Client's project requirements
            
        Returns:
            Generated content as string
        """
        print("🤖 Sending request to AI...")
        chain = self.prompt | self.llm
        response = chain.invoke({"requirement": requirement_text})
        return response.content

