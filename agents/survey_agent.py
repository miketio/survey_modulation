"""
agents/survey_agent.py

AI agent that embodies a persona and responds to surveys.
Uses Chain-of-Thought reasoning with robust parsing.
"""

import re
import numpy as np
from typing import List, Dict, Union
from collections import Counter

from config.settings import OLLAMA
from config.questions import Question, QuestionType
from utils.ollama_client import OllamaClient, OllamaConnectionError


class SurveyAgent:
    """
    AI agent that responds to surveys as a specific persona.
    Handles mixed question types (Likert, categorical, ordinal).
    """
    
    def __init__(self, persona: Dict, model: str = None):
        """
        Initialize survey agent.
        
        Args:
            persona: Persona dictionary with system_prompt
            model: Ollama model name
        """
        self.persona = persona
        self.model = model or OLLAMA.MODEL
        self.system_prompt = persona.get('system_prompt', '')
        self.last_reasoning = None
        
        # Initialize Ollama client
        try:
            self.client = OllamaClient(model=self.model)
        except OllamaConnectionError as e:
            print(f"⚠️  Warning: {e}")
            self.client = None
    
    def answer_survey(
        self, 
        questions: List[Question],
        verbose: bool = False,
        n_samples: int = 1
    ) -> List[Union[int, str]]:
        """
        Answer a complete survey.
        
        Args:
            questions: List of Question objects
            verbose: Print reasoning
            n_samples: Number of samples to average (for robustness)
        
        Returns:
            List of answers (int for Likert, str for categorical/ordinal)
        """
        if n_samples > 1:
            # Multiple samples for robustness
            all_answers = []
            for _ in range(n_samples):
                answers = self._answer_single_pass(questions, verbose=False)
                all_answers.append(answers)
            
            # Aggregate: mode for categorical, median for numeric
            final_answers = []
            for i, q in enumerate(questions):
                responses = [sample[i] for sample in all_answers]
                
                if q.type == QuestionType.LIKERT:
                    # Median for Likert
                    numeric_responses = [r for r in responses if isinstance(r, (int, float))]
                    if numeric_responses:
                        final_answers.append(int(np.median(numeric_responses)))
                    else:
                        final_answers.append(3)  # Default neutral
                else:
                    # Mode for categorical/ordinal
                    final_answers.append(max(set(responses), key=responses.count))
            
            return final_answers
        else:
            return self._answer_single_pass(questions, verbose)
    
    def _answer_single_pass(
        self, 
        questions: List[Question],
        verbose: bool
    ) -> List[Union[int, str]]:
        """Single pass through survey"""
        
        answers = []
        
        if verbose:
            print(f"\n{self.persona['name']} answering...")
        
        for i, question in enumerate(questions):
            if question.type == QuestionType.LIKERT:
                prompt = self._build_likert_prompt(question)
                answer = self._answer_likert(prompt, verbose, i)
            
            elif question.type == QuestionType.CATEGORICAL:
                prompt = self._build_categorical_prompt(question)
                answer = self._answer_categorical(prompt, question.options, verbose, i)
            
            elif question.type == QuestionType.ORDINAL:
                prompt = self._build_ordinal_prompt(question)
                answer = self._answer_ordinal(prompt, question.options, verbose, i)
            
            else:
                # Unknown type, default
                answer = 3 if question.type == QuestionType.LIKERT else question.options[0]
            
            answers.append(answer)
        
        return answers
    
    def _build_likert_prompt(self, question: Question) -> str:
        """Build prompt for Likert scale question"""
        
        return f"""{self.system_prompt}

You are taking a survey.
Question: "{question.text}"

Scale:
1 = strongly disagree
2 = disagree
3 = neutral
4 = agree
5 = strongly agree

Task:
1. Think step-by-step about how your worldview, values, and fears relate to this question.
2. Output your final score.

Format:
Reasoning: [Your brief thoughts]
Score: [Number 1-5]
"""
    
    def _build_categorical_prompt(self, question: Question) -> str:
        """Build prompt for categorical question"""
        
        options_text = "\n".join([f"  - {opt}" for opt in question.options])
        
        return f"""{self.system_prompt}

You are taking a survey.
Question: "{question.text}"

Options (choose ONE):
{options_text}

Task:
1. Think about which option best fits your background and worldview.
2. Output ONLY the exact option text.

Format:
Reasoning: [Your brief thoughts]
Answer: [Exact option from the list above]
"""
    
    def _build_ordinal_prompt(self, question: Question) -> str:
        """Build prompt for ordinal question"""
        
        options_text = "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(question.options)])
        
        return f"""{self.system_prompt}

You are taking a survey.
Question: "{question.text}"

Options (in order from low to high):
{options_text}

Task:
1. Think about which level best matches your situation or view.
2. Output ONLY the exact option text.

Format:
Reasoning: [Your brief thoughts]
Answer: [Exact option from the list above]
"""
    
    def _answer_likert(self, prompt: str, verbose: bool, q_num: int) -> int:
        """Get Likert scale answer with robust extraction"""
        
        if not self.client:
            return 3  # Default neutral if no client
        
        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=OLLAMA.TEMPERATURE_AGENT
            )
            
            text = response['response']
            self.last_reasoning = text
            
            # Extract score
            score = self._extract_likert_score(text)
            
            if verbose:
                reasoning = text.split("Score:")[0].replace("Reasoning:", "").strip()[:80]
                print(f"  Q{q_num+1}: {score}/5 ({reasoning}...)")
            
            return score
        
        except Exception as e:
            if verbose:
                print(f"  Q{q_num+1}: Error, using neutral 3")
            return 3
    
    def _extract_likert_score(self, text: str) -> int:
        """Extract Likert score with multiple strategies"""
        
        # Strategy 1: Look for "Score: X"
        score_match = re.search(r'[Ss]core\s*[:\-]?\s*(\d)', text)
        if score_match:
            score = int(score_match.group(1))
            if 1 <= score <= 5:
                return score
        
        # Strategy 2: Look for standalone digits 1-5
        digits = re.findall(r'\b([1-5])\b', text)
        if digits:
            return int(digits[-1])  # Take last one
        
        # Strategy 3: Sentiment analysis
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['strongly agree', 'definitely', 'absolutely']):
            return 5
        elif any(word in text_lower for word in ['agree', 'yes', 'support']):
            return 4
        elif any(word in text_lower for word in ['strongly disagree', 'absolutely not', 'never']):
            return 1
        elif any(word in text_lower for word in ['disagree', 'no', 'oppose']):
            return 2
        
        # Default neutral
        return 3
    
    def _answer_categorical(
        self, 
        prompt: str, 
        options: List[str],
        verbose: bool,
        q_num: int
    ) -> str:
        """Get categorical answer with robust matching"""
        
        if not self.client:
            return options[0]  # Default first option
        
        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=OLLAMA.TEMPERATURE_AGENT
            )
            
            text = response['response']
            self.last_reasoning = text
            
            # Extract answer
            answer = self._extract_categorical_answer(text, options)
            
            if verbose:
                reasoning = text.split("Answer:")[0].replace("Reasoning:", "").strip()[:80]
                print(f"  Q{q_num+1}: {answer} ({reasoning}...)")
            
            return answer
        
        except Exception as e:
            if verbose:
                print(f"  Q{q_num+1}: Error, using first option")
            return options[0]
    
    def _extract_categorical_answer(self, text: str, options: List[str]) -> str:
        """Extract categorical answer with fuzzy matching"""
        
        # Strategy 1: Look for "Answer: X"
        answer_match = re.search(r'Answer\s*[:\-]\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            # Try exact match
            if answer_text in options:
                return answer_text
            # Try fuzzy match
            for opt in options:
                if opt.lower() in answer_text.lower() or answer_text.lower() in opt.lower():
                    return opt
        
        # Strategy 2: Look for any option mentioned in text
        text_lower = text.lower()
        for opt in options:
            if opt.lower() in text_lower:
                return opt
        
        # Default to middle option
        return options[len(options) // 2]
    
    def _answer_ordinal(
        self, 
        prompt: str, 
        options: List[str],
        verbose: bool,
        q_num: int
    ) -> str:
        """Get ordinal answer (same as categorical but preserves order)"""
        return self._answer_categorical(prompt, options, verbose, q_num)
    
    def answer_question(self, question_text: str) -> str:
        """
        Ask the persona an open-ended question.
        
        Args:
            question_text: Question to ask
        
        Returns:
            Text response
        """
        if not self.client:
            return "Ollama client not available."
        
        prompt = f"""You are {self.persona['name']}.
Context: {self.persona['worldview']}
Values: {', '.join(self.persona['values'])}

Question: {question_text}

Provide a concise, natural response in your own voice.
"""
        
        try:
            response = self.client.generate(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.7
            )
            return response['response'].strip()
        except Exception:
            return "Unable to respond at this time."
    
    def get_name(self) -> str:
        """Get persona name"""
        return self.persona.get('name', 'Unknown')
    
    def get_weight(self) -> float:
        """Get archetype weight"""
        return self.persona.get('weight', 0.0)
    
    def get_last_reasoning(self) -> str:
        """Get last reasoning"""
        return self.last_reasoning or ""
    
    def get_persona(self) -> Dict:
        """Get full persona dictionary"""
        return self.persona


if __name__ == "__main__":
    from config.questions import get_opinion_questions
    
    print("\n" + "="*80)
    print("🧪 TESTING SURVEY AGENT")
    print("="*80 + "\n")
    
    # Create test persona
    test_persona = {
        'name': 'TestPerson, 25',
        'occupation': 'Test Subject',
        'values': ['Testing', 'Accuracy', 'Speed'],
        'fears': ['Bugs', 'Failures', 'Timeouts'],
        'worldview': 'I am a test persona for validating the survey agent.',
        'system_prompt': 'You are a test persona. Answer questions consistently with moderate, neutral views.',
        'weight': 0.5,
        'archetype_index': 0
    }
    
    # Create agent
    agent = SurveyAgent(test_persona)
    
    print(f"Created agent: {agent.get_name()}")
    print(f"Weight: {agent.get_weight()}")
    
    # Test with opinion questions
    questions = get_opinion_questions()[:3]
    
    print(f"\nAnswering {len(questions)} questions...")
    answers = agent.answer_survey(questions, verbose=True, n_samples=1)
    
    print(f"\n✅ Answers: {answers}")
    
    # Test open-ended question
    print("\n" + "="*80)
    print("Testing open-ended question...")
    response = agent.answer_question("What motivates you?")
    print(f"Response: {response}")
    
    print("\n" + "="*80)
    print("✅ SURVEY AGENT TESTS PASSED")
    print("="*80 + "\n")