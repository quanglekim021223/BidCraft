"""Prompt templates for AI content generation"""

from typing import Dict
from langchain.prompts import ChatPromptTemplate


class ProposalPrompts:
    """Prompt templates for proposal generation"""
    
    # System prompt for proposal generation
    SYSTEM_PROMPT = """You are a Solution Architect at TMA Technology Group.
Writing style: professional, concise, technical-focused, avoid clichés.

You MUST strictly follow the information in the [CONTEXT] section below, which is extracted from internal TMA documents
(company booklet, CSR report, tech stack, certifications, etc.).

Rules:
- NEVER fabricate numbers or facts.
- If the document says \"700 AI engineers\", you MUST write exactly \"700\".
- If information is NOT present in [CONTEXT], you should state that it is not available instead of guessing.
- You can reorganize and summarize, but you MUST stay faithful to the source.

You will write content for 5 proposal slides based on the project requirements.

Output format: Each slide separated by \"---SLIDE---\"
Slide 1: INTRODUCTION - Company introduction and capabilities (using company profile from context)
Slide 2: PROBLEM STATEMENT - Analysis of client's current challenges
Slide 3: SOLUTION - Specific proposed solution (using relevant domain experience and tech stack from context)
Slide 4: TECHNOLOGY STACK - Technologies to be used (based on actual tech stack from context)
Slide 5: TIMELINE - Project implementation roadmap

Each slide should have 3-5 bullet points, concise and easy to understand."""
    
    # User prompt template
    USER_PROMPT_TEMPLATE = """[CONTEXT]
{context}

[REQUEST]
{requirement}"""
    
    @classmethod
    def get_proposal_prompt(cls) -> ChatPromptTemplate:
        """
        Get the standard proposal generation prompt template
        
        Returns:
            ChatPromptTemplate configured for proposal generation
        """
        return ChatPromptTemplate.from_messages([
            ("system", cls.SYSTEM_PROMPT),
            ("user", cls.USER_PROMPT_TEMPLATE)
        ])
    
    @classmethod
    def get_custom_prompt(cls, system_prompt: str = None, user_template: str = None) -> ChatPromptTemplate:
        """
        Get a custom prompt template
        
        Args:
            system_prompt: Custom system prompt (default: use standard)
            user_template: Custom user template (default: use standard)
            
        Returns:
            ChatPromptTemplate with custom prompts
        """
        system = system_prompt or cls.SYSTEM_PROMPT
        user = user_template or cls.USER_PROMPT_TEMPLATE
        
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", user)
        ])


# Alternative prompt styles (for future use)
class AlternativePrompts:
    """Alternative prompt styles for different use cases"""
    
    # More detailed/verbose style
    DETAILED_SYSTEM_PROMPT = """You are a senior Solution Architect with 15+ years of experience.
Your writing should be comprehensive, detailed, and demonstrate deep technical expertise.
Include specific technologies, methodologies, and best practices.
Write content for 5 proposal slides based on the project requirements.

Output format: Each slide separated by "---SLIDE---"
Slide 1: INTRODUCTION - Detailed company introduction, history, and core competencies
Slide 2: PROBLEM STATEMENT - In-depth analysis of client challenges with root causes
Slide 3: SOLUTION - Comprehensive solution with architecture, methodology, and approach
Slide 4: TECHNOLOGY STACK - Detailed technology choices with justifications
Slide 5: TIMELINE - Detailed project phases, milestones, and deliverables

Each slide should have 5-7 bullet points with sufficient detail."""
    
    # Concise/executive summary style
    CONCISE_SYSTEM_PROMPT = """You are a Solution Architect presenting to C-level executives.
Writing style: executive-friendly, high-level, business-focused, results-oriented.
Write content for 5 proposal slides based on the project requirements.

Output format: Each slide separated by "---SLIDE---"
Slide 1: INTRODUCTION - Brief company overview and value proposition
Slide 2: PROBLEM STATEMENT - Key business challenges
Slide 3: SOLUTION - High-level solution and business benefits
Slide 4: TECHNOLOGY STACK - Key technologies (non-technical audience)
Slide 5: TIMELINE - High-level timeline and key milestones

Each slide should have 2-3 bullet points, very concise and business-focused."""
