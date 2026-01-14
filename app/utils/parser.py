"""Content parsing utilities"""

from typing import List, Optional


def parse_slides_content(
    ai_content: str, 
    expected_slides: int = 5,
    slide_titles: Optional[List[str]] = None
) -> List[str]:
    """
    Parse AI response content into separate slides
    
    Args:
        ai_content: Raw content from AI response
        expected_slides: Number of slides expected (default: 5)
        slide_titles: List of expected slide titles to remove if found in content
        
    Returns:
        List of slide contents
    """
    slides = ai_content.split("---SLIDE---")
    
    # Clean and format
    parsed = []
    for idx, slide in enumerate(slides):
        slide = slide.strip()
        if not slide:
            continue
            
        # Remove header if exists (e.g., "Slide 1: INTRODUCTION")
        lines = slide.split('\n')
        content_lines = []
        first_line_skipped = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip "Slide X:" lines
            if line.startswith("Slide"):
                continue
            
            # Remove title if it matches expected title (case-insensitive)
            if slide_titles and idx < len(slide_titles):
                expected_title = slide_titles[idx].strip()
                # Check if this line is exactly the title (case-insensitive)
                if line.upper() == expected_title.upper():
                    first_line_skipped = True
                    continue
                # Also check if line starts with title (e.g., "EXECUTIVE SUMMARY: ...")
                if line.upper().startswith(expected_title.upper() + ":"):
                    # Remove title prefix, keep the rest
                    line = line[len(expected_title) + 1:].strip()
                    if not line:
                        continue
            
            content_lines.append(line)
        
        if content_lines:
            parsed.append('\n'.join(content_lines))
    
    # Ensure we have expected number of slides
    while len(parsed) < expected_slides:
        parsed.append("Content is being updated...")
    
    return parsed[:expected_slides]

