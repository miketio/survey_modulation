"""
config/questions.py

Survey question definitions and question type management.
"""

from enum import Enum
from typing import List, Dict, Tuple
from dataclasses import dataclass

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
            result['scale'] = self.scale
        if self.options:
            result['options'] = self.options
            
        return result

# ============================================================================
# OPINION QUESTIONS (for archetypal analysis)
# ============================================================================

OPINION_QUESTIONS = [
    Question(
        id="Q1",
        text="I trust government institutions",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    ),
    Question(
        id="Q4",
        text="Technological changes bring more benefits than harm",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    ),
    Question(
        id="Q6",
        text="It is important to preserve traditional values",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    ),
    Question(
        id="Q8",
        text="Ecology is more important than economic growth",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    ),
    Question(
        id="Q9",
        text="I am willing to take risks for new opportunities",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    ),
]

# ============================================================================
# DEMOGRAPHIC QUESTIONS (for characterization)
# ============================================================================

DEMOGRAPHIC_QUESTIONS = [
    Question(
        id="D1",
        text="What is your age range?",
        type=QuestionType.ORDINAL,
        category="demographic",
        options=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    ),
    Question(
        id="D2",
        text="Where do you primarily live?",
        type=QuestionType.CATEGORICAL,
        category="demographic",
        options=["Urban/City", "Suburban", "Rural/Countryside"]
    ),
    Question(
        id="D3",
        text="What is your highest level of education?",
        type=QuestionType.ORDINAL,
        category="demographic",
        options=["High School", "Some College", "Bachelor's", "Master's", "PhD"]
    ),
    Question(
        id="D4",
        text="What is your political orientation?",
        type=QuestionType.CATEGORICAL,
        category="demographic",
        options=["Very Liberal", "Liberal", "Moderate", "Conservative", "Very Conservative"]
    ),
    Question(
        id="D5",
        text="How often do you follow the news?",
        type=QuestionType.ORDINAL,
        category="demographic",
        options=["Never", "Rarely", "Sometimes", "Often", "Daily"]
    ),
]

# ============================================================================
# SECOND SURVEY QUESTIONS (for validation)
# ============================================================================

SECOND_SURVEY_QUESTIONS = [
    Question(
        id="S1",
        text="Artificial intelligence threatens jobs",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    ),
    Question(
        id="S3",
        text="Government should regulate social media",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    ),
    Question(
        id="S5",
        text="Climate crisis requires immediate action",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    ),
    Question(
        id="S2",
        text="What is your income level?",
        type=QuestionType.ORDINAL,
        category="demographic",
        options=["<$30k", "$30k-$60k", "$60k-$100k", "$100k-$150k", ">$150k"]
    ),
    Question(
        id="S4",
        text="How do you commute to work/school?",
        type=QuestionType.CATEGORICAL,
        category="demographic",
        options=["Car", "Public Transit", "Bike/Walk", "Remote/No Commute"]
    ),
]

# ============================================================================
# COMBINED QUESTIONS
# ============================================================================

ALL_QUESTIONS = OPINION_QUESTIONS + DEMOGRAPHIC_QUESTIONS

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_opinion_questions() -> List[Question]:
    """Get only opinion questions (for archetypal analysis)"""
    return OPINION_QUESTIONS

def get_demographic_questions() -> List[Question]:
    """Get only demographic questions (for characterization)"""
    return DEMOGRAPHIC_QUESTIONS

def get_likert_questions(question_list: List[Question] = None) -> List[Question]:
    """Get only Likert scale questions"""
    if question_list is None:
        question_list = ALL_QUESTIONS
    return [q for q in question_list if q.type == QuestionType.LIKERT]

def get_categorical_questions(question_list: List[Question] = None) -> List[Question]:
    """Get categorical and ordinal questions"""
    if question_list is None:
        question_list = ALL_QUESTIONS
    return [q for q in question_list if q.type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]]

def questions_to_dict_list(questions: List[Question]) -> List[Dict]:
    """Convert list of Question objects to list of dicts"""
    return [q.to_dict() for q in questions]

def print_question_summary():
    """Print summary of question organization"""
    print("\n" + "="*60)
    print("QUESTION ORGANIZATION")
    print("="*60)
    
    print(f"\n📊 OPINION QUESTIONS (for archetypal analysis):")
    for q in OPINION_QUESTIONS:
        print(f"  • {q.id}: {q.text}")
    
    print(f"\n👤 DEMOGRAPHIC QUESTIONS (for characterization):")
    for q in DEMOGRAPHIC_QUESTIONS:
        print(f"  • {q.id}: {q.text}")
    
    print(f"\nTotal: {len(OPINION_QUESTIONS)} opinion + {len(DEMOGRAPHIC_QUESTIONS)} demographic")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_question_summary()
    
    # Test question creation
    print("\n🧪 Testing question validation...")
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
        q2 = Question(
            id="TEST2",
            text="Bad question",
            type=QuestionType.CATEGORICAL,
            category="demographic"
        )
    except ValueError as e:
        print(f"✅ Validation working: {e}")