"""
config/questions.py

Survey question definitions and question type management.
Now loads question data from JSON templates via loader.
"""

from enum import Enum
from typing import List, Dict, Tuple
from dataclasses import dataclass

from config.loader import load_questions as _load_questions

# ============================================================================
# QUESTION TYPES
# ============================================================================

class QuestionType(str, Enum):
    """Enumeration of supported question types"""
    
    LIKERT = "likert"           # 1-5 opinion scale
    CATEGORICAL = "categorical"  # Multiple choice
    ORDINAL = "ordinal"         # Ordered categories

# ============================================================================
# QUESTION SCHEMA
# ============================================================================

@dataclass
class Question:
    """Schema for a survey question"""
    
    id: str
    text: str
    type: QuestionType
    category: str  # 'opinion' or 'demographic'
    scale: Tuple[int, int] = None  # For Likert: (min, max)
    options: List[str] = None      # For categorical/ordinal
    
    def __post_init__(self):
        """Validate question configuration"""
        if self.type == QuestionType.LIKERT and not self.scale:
            self.scale = (1, 5)
        
        if self.type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]:
            if not self.options:
                raise ValueError(f"Question {self.id}: {self.type} requires options")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        result = {
            'id': self.id,
            'text': self.text,
            'type': self.type.value,
            'category': self.category,
        }
        
        if self.scale:
            result['scale'] = list(self.scale)
        if self.options:
            result['options'] = self.options
            
        return result

# ============================================================================
# QUESTION LOADING FUNCTIONS
# ============================================================================

def get_opinion_questions() -> List[Question]:
    """
    Get opinion questions (for archetypal analysis).
    Loads from JSON: data/config/questions/opinion_survey.json
    """
    return _load_questions("opinion_survey")

def get_demographic_questions() -> List[Question]:
    """
    Get demographic questions (for characterization).
    Loads from JSON: data/config/questions/demographics.json
    """
    return _load_questions("demographics")

def get_second_survey_questions() -> List[Question]:
    """
    Get second survey questions (for validation).
    Loads from JSON: data/config/questions/validation_survey.json
    """
    return _load_questions("validation_survey")

def get_all_questions() -> List[Question]:
    """Get all questions (opinion + demographic)"""
    return get_opinion_questions() + get_demographic_questions()

def load_custom_questions(template_name: str) -> List[Question]:
    """
    Load custom question template.
    
    Args:
        template_name: Name of template (without .json extension)
    
    Returns:
        List of Question objects
    """
    return _load_questions(template_name)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_likert_questions(question_list: List[Question] = None) -> List[Question]:
    """Get only Likert scale questions"""
    if question_list is None:
        question_list = get_all_questions()
    return [q for q in question_list if q.type == QuestionType.LIKERT]

def get_categorical_questions(question_list: List[Question] = None) -> List[Question]:
    """Get categorical and ordinal questions"""
    if question_list is None:
        question_list = get_all_questions()
    return [q for q in question_list if q.type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]]

def questions_to_dict_list(questions: List[Question]) -> List[Dict]:
    """Convert list of Question objects to list of dicts"""
    return [q.to_dict() for q in questions]

def get_available_templates() -> List[str]:
    """Get list of available question templates"""
    from config.loader import get_question_templates
    return get_question_templates()

def print_question_summary(template_name: str = None):
    """
    Print summary of question organization.
    
    Args:
        template_name: Specific template to summarize (default: all main templates)
    """
    print("\n" + "="*60)
    print("QUESTION ORGANIZATION")
    print("="*60)
    
    if template_name:
        questions = load_custom_questions(template_name)
        print(f"\n📊 {template_name.upper()}:")
        for q in questions:
            print(f"  • {q.id}: {q.text} ({q.type.value})")
    else:
        print(f"\n📊 OPINION QUESTIONS (for archetypal analysis):")
        for q in get_opinion_questions():
            print(f"  • {q.id}: {q.text}")
        
        print(f"\n👤 DEMOGRAPHIC QUESTIONS (for characterization):")
        for q in get_demographic_questions():
            print(f"  • {q.id}: {q.text}")
        
        print(f"\n✅ VALIDATION QUESTIONS (for second survey):")
        for q in get_second_survey_questions():
            print(f"  • {q.id}: {q.text}")
    
    print("="*60 + "\n")

def print_available_templates():
    """Print all available question templates"""
    print("\n" + "="*60)
    print("AVAILABLE QUESTION TEMPLATES")
    print("="*60 + "\n")
    
    templates = get_available_templates()
    
    if not templates:
        print("No templates found. Run 'python migrate_to_json.py' to create default templates.")
    else:
        for template in templates:
            questions = load_custom_questions(template)
            print(f"📋 {template}")
            print(f"   Questions: {len(questions)}")
            
            # Count by type
            likert = sum(1 for q in questions if q.type == QuestionType.LIKERT)
            categorical = sum(1 for q in questions if q.type == QuestionType.CATEGORICAL)
            ordinal = sum(1 for q in questions if q.type == QuestionType.ORDINAL)
            
            print(f"   Likert: {likert} | Categorical: {categorical} | Ordinal: {ordinal}")
            print()
    
    print("="*60 + "\n")

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING QUESTION LOADER")
    print("="*80 + "\n")
    
    # Print available templates
    print_available_templates()
    
    # Test loading opinion questions
    print("Testing opinion questions...")
    try:
        questions = get_opinion_questions()
        print(f"✅ Loaded {len(questions)} opinion questions")
        for q in questions[:3]:
            print(f"   • {q.id}: {q.text}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   Run 'python migrate_to_json.py' to create default templates")
    
    # Test loading demographic questions
    print("\nTesting demographic questions...")
    try:
        questions = get_demographic_questions()
        print(f"✅ Loaded {len(questions)} demographic questions")
        for q in questions[:3]:
            print(f"   • {q.id}: {q.text}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
    
    # Test question validation
    print("\n" + "="*80)
    print("Testing question validation...")
    try:
        # This should work
        q1 = Question(
            id="TEST1",
            text="Test question",
            type=QuestionType.LIKERT,
            category="opinion"
        )
        print(f"✅ Created: {q1.id}")
        
        # This should fail (categorical without options)
        try:
            q2 = Question(
                id="TEST2",
                text="Bad question",
                type=QuestionType.CATEGORICAL,
                category="demographic"
            )
        except ValueError as e:
            print(f"✅ Validation working: {e}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n" + "="*80)
    print("✅ QUESTION LOADER TESTS COMPLETE")
    print("="*80 + "\n")