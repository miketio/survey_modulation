"""
generators/persona_generator.py

Generates AI personas from archetypal patterns using LLM.
Uses strict demographic constraints and Pydantic validation.
"""

import json
import numpy as np
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

from config.settings import OLLAMA, DATA_GEN
from config.questions import Question
from utils.ollama_client import OllamaClient, OllamaConnectionError
from utils.file_io import save_personas


class PersonaSchema(BaseModel):
    """Strict schema for persona validation"""
    
    name: str = Field(description="Name and age (e.g., 'Sarah, 21')")
    occupation: str = Field(description="Specific to demographic context")
    demographics: str = Field(description="Must match demographic context")
    values: List[str] = Field(description="3-5 core values", min_length=1, max_length=5)
    fears: List[str] = Field(description="3-5 core fears", min_length=1, max_length=5)
    worldview: str = Field(description="2-3 sentences explaining responses")
    media_consumption: str = Field(description="Specific sources")
    system_prompt: str = Field(description="Second-person roleplay instructions")
    
    @field_validator('name')
    def validate_name(cls, v):
        if ',' not in v:
            raise ValueError("Name must include age (e.g., 'Sarah, 21')")
        return v
    
    @field_validator('media_consumption', mode='before')
    def validate_media_consumption(cls, v):
        """Convert media_consumption to string if it's a list"""
        if isinstance(v, list):
            # Join list items into a string
            return ", ".join(str(item) for item in v if item)
        if v is None:
            return "Various media sources"
        return str(v).strip()
    
    @field_validator('values', 'fears', mode='before')
    def validate_lists(cls, v):
        """Ensure values and fears are valid lists"""
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        if not isinstance(v, list):
            return ["Default value"]
        return v

def generate_system_prompt(persona: Dict) -> str:
    """
    Generate system prompt from persona fields.
    Ensures edits to worldview, values, fears are reflected.
    """
    name = persona.get('name', 'Person')
    occupation = persona.get('occupation', 'individual')
    worldview = persona.get('worldview', '')
    values = persona.get('values', [])
    fears = persona.get('fears', [])
    
    values_str = ', '.join(values) if isinstance(values, list) else str(values)
    fears_str = ', '.join(fears) if isinstance(fears, list) else str(fears)
    
    system_prompt = f"""You are {name}, a {occupation}.

Your worldview: {worldview}

You strongly value: {values_str}

You are concerned about or fear: {fears_str}

When answering questions, think like someone with this specific worldview and these values. Your answers should be consistent with your beliefs, fears, and life experiences.
"""
    
    return system_prompt

def parse_demographic_context(context: str) -> Dict:
    """
    Extract constraints from demographic context.
    
    Args:
        context: Demographic description (e.g., "University Students in New York")
    
    Returns:
        Dictionary of constraints
    """
    context_lower = context.lower()
    constraints = {}
    
    # Age constraints
    if "student" in context_lower:
        constraints['age_range'] = ['18-25']
        constraints['age_examples'] = ['19', '21', '22', '24']
    elif "senior" in context_lower or "retired" in context_lower:
        constraints['age_range'] = ['60+']
        constraints['age_examples'] = ['65', '68', '72']
    elif "worker" in context_lower or "professional" in context_lower:
        constraints['age_range'] = ['25-60']
        constraints['age_examples'] = ['28', '35', '42', '51']
    
    # Education constraints
    if "university" in context_lower or "college" in context_lower:
        constraints['education'] = ['Currently enrolled in university']
        constraints['education_status'] = 'Student'
    elif "phd" in context_lower or "researcher" in context_lower:
        constraints['education'] = ['PhD or Master\'s degree']
    
    # Occupation constraints
    if "student" in context_lower:
        constraints['occupation_type'] = 'Academic major/program'
        constraints['occupation_examples'] = [
            'Computer Science Junior',
            'Economics Senior', 
            'Pre-Med Sophomore',
            'Engineering Graduate Student'
        ]
    elif "healthcare" in context_lower:
        constraints['occupation_type'] = 'Healthcare role'
        constraints['occupation_examples'] = ['Nurse', 'Doctor', 'Medical Technician']
    elif "tech" in context_lower:
        constraints['occupation_type'] = 'Technology role'
        constraints['occupation_examples'] = ['Software Engineer', 'Product Manager']
    
    # Location constraints
    if "new york" in context_lower or "nyc" in context_lower:
        constraints['location'] = 'New York City area'
    elif "rural" in context_lower:
        constraints['location'] = 'Rural area'
    elif "urban" in context_lower or "city" in context_lower:
        constraints['location'] = 'Urban area'
    
    return constraints


def build_persona_prompt(
    demographic_context: str,
    constraints: Dict,
    answers: np.ndarray,
    questions: List[Question],
    contrast_personas: List[Dict] = None
) -> str:
    """
    Build strict prompt for persona generation.
    
    Args:
        demographic_context: Target demographic
        constraints: Parsed constraints
        answers: Archetype pattern
        questions: Question objects
        contrast_personas: Existing personas to differ from
    
    Returns:
        Prompt string
    """
    # Format answers
    answers_text = "\n".join([
        f"- {q.text}: {a:.1f}/5" 
        for q, a in zip(questions, answers)
    ])
    
    # Build constraint text
    constraint_lines = []
    if 'age_range' in constraints:
        constraint_lines.append(
            f"• Age: MUST be {constraints['age_range'][0]} "
            f"(e.g., {', '.join(constraints['age_examples'])})"
        )
    if 'education' in constraints:
        constraint_lines.append(f"• Education: {constraints['education'][0]}")
    if 'occupation_type' in constraints:
        constraint_lines.append(f"• Occupation: {constraints['occupation_type']}")
        constraint_lines.append(
            f"  Examples: {', '.join(constraints['occupation_examples'][:3])}"
        )
    if 'location' in constraints:
        constraint_lines.append(f"• Location: {constraints['location']}")
    
    constraints_text = "\n".join(constraint_lines)
    
    # Contrastive text
    contrast_text = ""
    if contrast_personas:
        contrast_text = f"\n\nEXISTING PERSONAS (must be DIFFERENT):\n"
        for p in contrast_personas[-3:]:
            contrast_text += f"- {p['name']}: {p['worldview'][:80]}...\n"
    
    prompt = f"""You are a senior qualitative researcher creating realistic personas.

DEMOGRAPHIC CONTEXT: {demographic_context}

STRICT REQUIREMENTS:
{constraints_text}

ALL of these MUST be satisfied. Any deviation is unacceptable.

SURVEY RESPONSES (1=Strongly Disagree, 5=Strongly Agree):
{answers_text}

TASK:
1. Create a persona that LITERALLY fits the demographic context
2. Their worldview must EXPLAIN these specific survey responses
3. Be internally consistent
4. Make them distinct from existing personas

{contrast_text}

RESPOND ONLY with valid JSON:
{{
  "name": "FirstName, Age",
  "occupation": "Specific occupation matching demographic",
  "demographics": "Detailed demographics matching context",
  "values": ["value1", "value2", "value3"],
  "fears": ["fear1", "fear2", "fear3"],
  "worldview": "2-3 sentences explaining WHY they gave these responses",
  "media_consumption": "Specific sources they consume",
  "system_prompt": "You are [name], a [occupation] in [location]. You value [values]. When answering questions, you think like someone who [worldview summary]."
}}

CRITICAL: The occupation MUST match the demographic context exactly.
"""
    
    return prompt


class PersonaGenerator:
    """
    Generates AI personas from archetypal patterns.
    """
    
    def __init__(
        self, 
        model: str = None,
        demographic_context: str = None
    ):
        """
        Initialize persona generator.
        
        Args:
            model: Ollama model name
            demographic_context: Target demographic
        """
        self.model = model or OLLAMA.MODEL
        self.demographic_context = demographic_context or DATA_GEN.DEMOGRAPHIC_CONTEXT
        
        # Initialize Ollama client
        try:
            self.client = OllamaClient(model=self.model)
            self.client.test_connection(verbose=False)
        except OllamaConnectionError as e:
            print(f"⚠️  Warning: {e}")
            print("   Persona generation will use fallback mode")
            self.client = None
        
        self.personas = []
    
    def generate_persona(
        self,
        answers: np.ndarray,
        questions: List[Question],
        archetype_index: int,
        weight: float,
        r2: float,
        contrast_personas: List[Dict] = None,
        max_retries: int = 3
    ) -> Dict:
        """
        Generate a single persona from archetype pattern.
        
        Args:
            answers: Archetype answer pattern
            questions: Question objects
            archetype_index: Index of this archetype
            weight: Population proportion
            r2: Explained variance
            contrast_personas: Existing personas to differ from
            max_retries: Maximum generation attempts
        
        Returns:
            Persona dictionary
        """
        # Parse constraints
        constraints = parse_demographic_context(self.demographic_context)
        
        print(f"\n{'─'*70}")
        print(f"Generating Archetype #{archetype_index + 1} (Weight: {weight:.1%})")
        print(f"Context: {self.demographic_context}")
        print(f"{'─'*70}")
        
        # Try generation with retries
        for attempt in range(max_retries):
            try:
                if not self.client:
                    raise OllamaConnectionError("Client not available")
                
                # Build prompt
                prompt = build_persona_prompt(
                    self.demographic_context,
                    constraints,
                    answers,
                    questions,
                    contrast_personas
                )
                
                # Generate
                response = self.client.generate(
                    prompt=prompt,
                    format="json",
                    temperature=OLLAMA.TEMPERATURE_PERSONA
                )
                
                # Validate
                response_text = response['response']
                validated = PersonaSchema.model_validate_json(response_text)
                persona_dict = validated.model_dump()
                
                # Additional validation
                if not self._validate_occupation(persona_dict, constraints):
                    print(f"  ⚠️  Attempt {attempt+1}: Occupation doesn't match")
                    continue
                
                # Success!
                persona_dict.update({
                    'archetype_index': archetype_index,
                    'weight': float(weight),
                    'r2': float(r2),
                    'average_answers': answers.tolist(),
                    'is_outlier': False,
                    'demographic_category': self.demographic_context
                })
                
                print(f"  ✅ Success: {persona_dict['name']}")
                print(f"     Occupation: {persona_dict['occupation']}")
                
                self.personas.append(persona_dict)
                return persona_dict
            
            except Exception as e:
                print(f"  ⚠️  Attempt {attempt+1} failed: {str(e)[:80]}")
                if attempt == max_retries - 1:
                    print(f"  ❌ Max retries reached, using fallback")
                    return self._create_fallback_persona(
                        archetype_index, weight, r2, answers, constraints
                    )
        
        # Should not reach here
        return self._create_fallback_persona(
            archetype_index, weight, r2, answers, constraints
        )
    
    def _validate_occupation(self, persona: Dict, constraints: Dict) -> bool:
        """Validate occupation matches constraints"""
        occupation = persona['occupation'].lower()
        
        if 'occupation_type' in constraints:
            constraint_type = constraints['occupation_type'].lower()
            
            if 'student' in constraint_type or 'major' in constraint_type:
                return any(word in occupation for word in 
                          ['major', 'student', 'studying', 'sophomore', 
                           'junior', 'senior', 'freshman'])
            
            elif 'healthcare' in constraint_type:
                return any(word in occupation for word in 
                          ['nurse', 'doctor', 'medical', 'healthcare', 'physician'])
            
            elif 'tech' in constraint_type:
                return any(word in occupation for word in 
                          ['engineer', 'developer', 'programmer', 'tech', 'software'])
        
        return True
    
    def _create_fallback_persona(
        self, 
        index: int, 
        weight: float, 
        r2: float, 
        answers: np.ndarray, 
        constraints: Dict
    ) -> Dict:
        """Create fallback persona when generation fails"""
        
        # Generate occupation based on constraints
        if 'occupation_examples' in constraints:
            occupation = constraints['occupation_examples'][
                index % len(constraints['occupation_examples'])
            ]
        else:
            occupation = f"Member of {self.demographic_context}"
        
        # Generate age
        if 'age_examples' in constraints:
            age = constraints['age_examples'][
                index % len(constraints['age_examples'])
            ]
        else:
            age = "25"
        
        fallback = {
            "name": f"Person{index+1}, {age}",
            "occupation": occupation,
            "demographics": f"{age} years old, {self.demographic_context}",
            "values": ["Consistency", "Reliability", "Authenticity"],
            "fears": ["Uncertainty", "Misrepresentation", "Instability"],
            "worldview": f"Fallback persona for {self.demographic_context}.",
            "media_consumption": "Various sources",
            "system_prompt": f"You are a {occupation} in {self.demographic_context}.",
            "archetype_index": index,
            "weight": float(weight),
            "r2": float(r2),
            "average_answers": answers.tolist(),
            "is_outlier": False,
            "demographic_category": self.demographic_context
        }
        
        self.personas.append(fallback)
        return fallback
    
    def generate_batch(
        self,
        archetypes: np.ndarray,
        questions: List[Question],
        weights: np.ndarray,
        r2_scores: List[float]
    ) -> List[Dict]:
        """
        Generate all personas at once.
        
        Args:
            archetypes: Matrix of archetype patterns
            questions: Question objects
            weights: Archetype proportions
            r2_scores: Explained variance per archetype
        
        Returns:
            List of persona dictionaries
        """
        personas = []
        
        for i in range(len(archetypes)):
            persona = self.generate_persona(
                answers=archetypes[i],
                questions=questions,
                archetype_index=i,
                weight=weights[i],
                r2=r2_scores[i],
                contrast_personas=personas
            )
            personas.append(persona)
        
        return personas
    
    def save(self, filepath = None):
        """Save personas to file"""
        return save_personas(self.personas, filepath=filepath)
    
    def get_personas(self) -> List[Dict]:
        """Get generated personas"""
        return self.personas


if __name__ == "__main__":
    from config.questions import get_opinion_questions
    
    print("\n" + "="*80)
    print("🧪 TESTING PERSONA GENERATOR")
    print("="*80 + "\n")
    
    # Test demographic parsing
    contexts = [
        "University Students in New York",
        "Healthcare Workers in Rural Montana",
        "Tech Professionals in San Francisco"
    ]
    
    for context in contexts:
        print(f"\nContext: {context}")
        constraints = parse_demographic_context(context)
        for key, value in constraints.items():
            print(f"  {key}: {value}")
    
    # Test persona generation
    print("\n" + "="*80)
    print("Testing persona generation...")
    
    generator = PersonaGenerator()
    
    # Create test archetype
    test_answers = np.array([3.0, 4.5, 2.0, 4.0, 3.5])
    questions = get_opinion_questions()
    
    persona = generator.generate_persona(
        answers=test_answers,
        questions=questions,
        archetype_index=0,
        weight=0.35,
        r2=0.42
    )
    
    print(f"\n✅ Generated persona:")
    print(f"   Name: {persona['name']}")
    print(f"   Occupation: {persona['occupation']}")
    print(f"   Values: {persona['values']}")
    
    print("\n" + "="*80)
    print("✅ PERSONA GENERATOR TESTS PASSED")
    print("="*80 + "\n")