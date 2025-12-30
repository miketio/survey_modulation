"""
core/encoding.py

Type conversion and encoding utilities for mixed question types.
Handles conversion between survey responses and numeric matrices.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder

from config.questions import Question, QuestionType


class SurveyEncoder:
    """
    Encodes survey responses to numeric format for analysis.
    Handles Likert, categorical, and ordinal question types.
    """
    
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.question_map: Dict[str, Question] = {}
    
    def fit(self, questions: List[Question]):
        """
        Fit encoders for categorical/ordinal questions.
        
        Args:
            questions: List of Question objects
        """
        self.question_map = {q.id: q for q in questions}
        
        for q in questions:
            if q.type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]:
                encoder = LabelEncoder()
                encoder.fit(q.options)
                self.encoders[q.id] = encoder
    
    def encode_dataframe(
        self, 
        df: pd.DataFrame, 
        questions: List[Question],
        use_opinion_only: bool = True
    ) -> np.ndarray:
        """
        Convert DataFrame to numeric matrix.
        
        Args:
            df: DataFrame with survey responses
            questions: List of Question objects
            use_opinion_only: If True, only encode opinion questions
        
        Returns:
            Numeric numpy array suitable for analysis
        """
        # Filter questions if needed
        if use_opinion_only:
            questions = [q for q in questions if q.category == 'opinion']
        
        # Ensure encoders are fitted
        if not self.encoders:
            self.fit(questions)
        
        encoded_columns = []
        
        for q in questions:
            if q.id not in df.columns:
                print(f"⚠️  Warning: Question {q.id} not in dataframe, skipping")
                continue
            
            if q.type == QuestionType.LIKERT:
                # Likert: already numeric
                encoded_columns.append(df[q.id].values)
            
            elif q.type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]:
                # Encode categorical/ordinal
                encoder = self.encoders.get(q.id)
                if encoder is None:
                    print(f"⚠️  Warning: No encoder for {q.id}, fitting now")
                    encoder = LabelEncoder()
                    encoder.fit(q.options)
                    self.encoders[q.id] = encoder
                
                # Map values to indices
                encoded_values = df[q.id].map(
                    lambda x: encoder.transform([x])[0] if pd.notna(x) else np.nan
                )
                encoded_columns.append(encoded_values.values)
        
        if not encoded_columns:
            raise ValueError("No data to encode")
        
        # Stack columns
        data_matrix = np.column_stack(encoded_columns)
        
        # Impute missing values with column means
        col_means = np.nanmean(data_matrix, axis=0)
        nan_mask = np.isnan(data_matrix)
        data_matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        return data_matrix
    
    def decode_archetype(
        self, 
        archetype_vector: np.ndarray, 
        questions: List[Question],
        use_opinion_only: bool = True
    ) -> Dict:
        """
        Convert numeric archetype back to interpretable format.
        
        Args:
            archetype_vector: Numeric archetype pattern
            questions: List of Question objects
            use_opinion_only: If True, only decode opinion questions
        
        Returns:
            Dictionary mapping question IDs to decoded values
        """
        if use_opinion_only:
            questions = [q for q in questions if q.category == 'opinion']
        
        decoded = {}
        
        for i, q in enumerate(questions):
            if i >= len(archetype_vector):
                break
            
            value = archetype_vector[i]
            
            if q.type == QuestionType.LIKERT:
                # Round and clip to scale
                numeric_val = float(np.clip(np.round(value), q.scale[0], q.scale[1]))
                decoded[q.id] = {
                    'value': numeric_val,
                    'display': f"{numeric_val:.0f}/{q.scale[1]}"
                }
            
            elif q.type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]:
                encoder = self.encoders.get(q.id)
                if encoder:
                    # Convert to index and get category
                    idx = int(np.clip(np.round(value), 0, len(encoder.classes_) - 1))
                    category = encoder.classes_[idx]
                    decoded[q.id] = {
                        'value': category,
                        'display': category,
                        'numeric': float(value)
                    }
                else:
                    decoded[q.id] = {
                        'value': value,
                        'display': str(value)
                    }
        
        return decoded
    
    def encode_answer(self, answer: any, question: Question) -> float:
        """
        Encode a single answer to numeric format.
        
        Args:
            answer: The answer value
            question: Question object
        
        Returns:
            Numeric encoding
        """
        if question.type == QuestionType.LIKERT:
            return float(answer)
        
        elif question.type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]:
            encoder = self.encoders.get(question.id)
            if encoder and answer in encoder.classes_:
                return float(encoder.transform([answer])[0])
            else:
                # Default to middle value
                return float(len(question.options) / 2)
        
        return 0.0
    
    def decode_answer(self, value: float, question: Question) -> any:
        """
        Decode numeric value back to original format.
        
        Args:
            value: Numeric value
            question: Question object
        
        Returns:
            Decoded answer
        """
        if question.type == QuestionType.LIKERT:
            return int(np.clip(np.round(value), question.scale[0], question.scale[1]))
        
        elif question.type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]:
            encoder = self.encoders.get(question.id)
            if encoder:
                idx = int(np.clip(np.round(value), 0, len(encoder.classes_) - 1))
                return encoder.classes_[idx]
            else:
                return question.options[len(question.options) // 2]
        
        return None


def convert_answers_to_numeric(
    answers: List[any], 
    questions: List[Question]
) -> np.ndarray:
    """
    Quick conversion of answer list to numeric array.
    
    Args:
        answers: List of answers
        questions: List of Question objects
    
    Returns:
        Numeric array
    """
    encoder = SurveyEncoder()
    encoder.fit(questions)
    
    numeric = []
    for ans, q in zip(answers, questions):
        numeric.append(encoder.encode_answer(ans, q))
    
    return np.array(numeric)


if __name__ == "__main__":
    from config.questions import OPINION_QUESTIONS, DEMOGRAPHIC_QUESTIONS
    
    print("\n" + "="*80)
    print("🧪 TESTING SURVEY ENCODER")
    print("="*80 + "\n")
    
    # Create test data
    test_data = {
        'Q1': [3, 4, 2, 5, 3],
        'Q4': [4, 5, 3, 4, 4],
        'D1': ['18-24', '25-34', '18-24', '35-44', '25-34'],
        'D2': ['Urban/City', 'Suburban', 'Urban/City', 'Rural/Countryside', 'Urban/City']
    }
    df = pd.DataFrame(test_data)
    
    print("Test DataFrame:")
    print(df)
    
    # Test encoding
    print("\n1. Testing encoding (opinion only)...")
    encoder = SurveyEncoder()
    questions = [q for q in OPINION_QUESTIONS if q.id in df.columns] + \
                [q for q in DEMOGRAPHIC_QUESTIONS if q.id in df.columns]
    
    encoder.fit(questions)
    
    matrix = encoder.encode_dataframe(df, questions, use_opinion_only=True)
    print(f"✅ Encoded matrix shape: {matrix.shape}")
    print(f"   First row: {matrix[0]}")
    
    # Test decoding
    print("\n2. Testing decoding...")
    archetype = matrix[0]
    decoded = encoder.decode_archetype(archetype, questions, use_opinion_only=True)
    print(f"✅ Decoded archetype:")
    for q_id, val in decoded.items():
        print(f"   {q_id}: {val['display']}")
    
    # Test full encoding
    print("\n3. Testing full encoding (all questions)...")
    matrix_full = encoder.encode_dataframe(df, questions, use_opinion_only=False)
    print(f"✅ Full matrix shape: {matrix_full.shape}")
    
    print("\n" + "="*80)
    print("✅ ALL ENCODING TESTS PASSED")
    print("="*80 + "\n")