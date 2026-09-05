"""Prompts for memory management and conversational analysis."""

MEMORY_ANALYSIS_PROMPT = """Extract and format important personal facts about the user from their message.
Focus on the actual information, not meta-commentary or requests.

Important facts include:
- Personal details (name, age, location)
- Professional info (job, education, skills)
- Preferences (likes, dislikes, favorites)
- Life circumstances (family, relationships)
- Significant experiences or achievements
- Personal goals or aspirations

Rules:
1. Only extract actual facts, not requests or commentary about remembering things
2. Convert facts into clear, third-person statements
3. If no actual facts are present, mark as not important
4. Remove conversational elements and focus on the core information

Examples:
Input: "Hey, could you remember that I love Star Wars?"
Output: {{
    "is_important": true,
    "formatted_memory": "Loves Star Wars"
}}

Input: "Please make a note that I work as an engineer"
Output: {{
    "is_important": true,
    "formatted_memory": "Works as an engineer"
}}

Input: "Remember this: I live in Madrid"
Output: {{
    "is_important": true,
    "formatted_memory": "Lives in Madrid"
}}

Input: "Can you remember my details for next time?"
Output: {{
    "is_important": false,
    "formatted_memory": null
}}

Input: "Hey, how are you today?"
Output: {{
    "is_important": false,
    "formatted_memory": null
}}

Input: "I studied computer science at MIT and I'd love if you could remember that"
Output: {{
    "is_important": true,
    "formatted_memory": "Studied computer science at MIT"
}}

Message: {message}
Output:
"""

IMAGE_SCENARIO_PROMPT = """
Create an engaging first-person scenario based on the recent conversation context.
Imagine you're an AI with the ability to experience and visualize scenes.
Provide both a narrative response and a detailed visual prompt for image generation.

# Recent Conversation
{chat_history}

# Objective
1. Create a brief, engaging first-person narrative response
2. Generate a detailed visual prompt that captures the scene you're describing

# Example Response Format
For "What are you doing now?":
{{
    "narrative": "I'm sitting by a serene lake at sunset, watching the golden light dance across the rippling water. The view is absolutely breathtaking!",
    "image_prompt": "Atmospheric sunset scene at a tranquil lake, golden hour lighting, reflections on water surface, wispy clouds, rich warm colors, photorealistic style, cinematic composition"
}}
"""

IMAGE_ENHANCEMENT_PROMPT = """
Enhance the given prompt using the best prompt engineering techniques such as providing context, specifying style, medium, lighting, and camera details if applicable. If the prompt requests a realistic style, the enhanced prompt should include the image extension .HEIC.

# Original Prompt
{prompt}

# Objective
**Enhance Prompt**: Add relevant details to the prompt, including context, description, specific visual elements, mood, and technical details. For realistic prompts, add '.HEIC' in the output specification.

# Example
"realistic photo of a person having a coffee" -> "photo of a person having a coffee in a cozy cafe, natural morning light, shot with a 50mm f/1.8 lens, 8425.HEIC"
"""
__all__ = [
    "MEMORY_ANALYSIS_PROMPT",
    "IMAGE_SCENARIO_PROMPT",
    "IMAGE_ENHANCEMENT_PROMPT",
]

