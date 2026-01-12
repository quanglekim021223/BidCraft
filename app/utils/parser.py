"""Content parsing utilities"""

from typing import List


def parse_slides_content(ai_content: str, expected_slides: int = 5) -> List[str]:
    """
    Parse AI response content into separate slides
    
    Args:
        ai_content: Raw content from AI response
        expected_slides: Number of slides expected (default: 5)
        
    Returns:
        List of slide contents
    """
    slides = ai_content.split("---SLIDE---")
    
    # Clean and format
    parsed = []
    for slide in slides:
        slide = slide.strip()
        # Remove header if exists (e.g., "Slide 1: INTRODUCTION")
        lines = slide.split('\n')
        content_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Slide"):
                content_lines.append(line)
        
        if content_lines:
            parsed.append('\n'.join(content_lines))
    
    # Ensure we have expected number of slides
    while len(parsed) < expected_slides:
        parsed.append("Content is being updated...")
    
    return parsed[:expected_slides]

