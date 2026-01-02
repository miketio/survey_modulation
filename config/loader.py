"""
config/loader.py

Unified configuration loader for Survey Archetypes system.
Provides single source of truth for questions, archetypes, and settings.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from functools import lru_cache
import jsonschema
from dataclasses import asdict

# Prevent circular import by only importing for type checking
if TYPE_CHECKING:
    from config.questions import Question, QuestionType

class ConfigLoader:
    """
    Central configuration loader with validation and caching.
    Ensures consistency between Python CLI and Web App.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_dir: Root config directory (default: data/config/)
        """
        if config_dir is None:
            base_dir = Path(__file__).parent.parent
            config_dir = base_dir / "data" / "config"
        
        self.config_dir = Path(config_dir)
        self.questions_dir = self.config_dir / "questions"
        self.archetypes_dir = self.config_dir / "archetypes"
        self.personas_dir = self.config_dir / "personas"
        self.schema_dir = self.config_dir / "schema"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Cache
        self._cache = {}
    
    def _ensure_directories(self):
        """Create directory structure if it doesn't exist"""
        for dir_path in [
            self.config_dir,
            self.questions_dir,
            self.archetypes_dir,
            self.personas_dir,
            self.schema_dir,
            self.archetypes_dir / "discovered"
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # QUESTIONS
    # ============================================================================
    
    def load_questions(self, template_name: str = "opinion_survey") -> List['Question']:
        """
        Load questions from JSON template.
        
        Args:
            template_name: Name of template (without .json)
        
        Returns:
            List of Question objects
        """
        # Local import to prevent circular dependency
        from config.questions import Question, QuestionType

        cache_key = f"questions_{template_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        filepath = self.questions_dir / f"{template_name}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Question template '{template_name}' not found at {filepath}"
            )
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate schema if available
        self._validate_if_schema_exists(data, "question_schema.json")
        
        # Convert to Question objects
        questions = []
        for q_data in data.get('questions', []):
            q_type = QuestionType(q_data['type'])
            
            question = Question(
                id=q_data['id'],
                text=q_data['text'],
                type=q_type,
                category=q_data.get('category', 'opinion'),
                scale=tuple(q_data['scale']) if 'scale' in q_data else None,
                options=q_data.get('options')
            )
            questions.append(question)
        
        self._cache[cache_key] = questions
        return questions
    
    def save_questions(
        self, 
        questions: List['Question'], 
        name: str,
        description: str = "",
        version: str = "1.0"
    ) -> Path:
        """
        Save questions as JSON template.
        
        Args:
            questions: List of Question objects
            name: Template name (without .json)
            description: Template description
            version: Template version
        
        Returns:
            Path to saved file
        """
        filepath = self.questions_dir / f"{name}.json"
        
        data = {
            "name": name,
            "description": description,
            "version": version,
            "questions": [self._question_to_dict(q) for q in questions]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        cache_key = f"questions_{name}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        return filepath
    
    def get_question_templates(self) -> List[str]:
        """Get list of available question templates"""
        if not self.questions_dir.exists():
            return []
        
        templates = [
            f.stem for f in self.questions_dir.glob("*.json")
        ]
        return sorted(templates)
    
    def _question_to_dict(self, question: 'Question') -> Dict:
        """Convert Question object to dict for JSON"""
        result = {
            'id': question.id,
            'text': question.text,
            'type': question.type.value,
            'category': question.category
        }
        
        if question.scale:
            result['scale'] = list(question.scale)
        
        if question.options:
            result['options'] = question.options
        
        return result
    
    # ============================================================================
    # ARCHETYPES
    # ============================================================================
    
    def load_archetypes(self, name: str = "default") -> List[Dict]:
        """
        Load archetype definitions from JSON.
        
        Args:
            name: Archetype set name (without .json)
        
        Returns:
            List of archetype dictionaries
        """
        cache_key = f"archetypes_{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try regular archetypes first
        filepath = self.archetypes_dir / f"{name}.json"
        
        # If not found, try discovered archetypes
        if not filepath.exists():
            filepath = self.archetypes_dir / "discovered" / f"{name}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Archetype set '{name}' not found at {filepath}"
            )
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate schema if available
        self._validate_if_schema_exists(data, "archetype_schema.json")
        
        archetypes = data.get('archetypes', [])
        self._cache[cache_key] = archetypes
        return archetypes
    
    def save_archetypes(
        self,
        archetypes: List[Dict],
        name: str,
        description: str = "",
        version: str = "1.0",
        demographic_context: str = "",
        discovered: bool = False
    ) -> Path:
        """
        Save archetype definitions as JSON.
        
        Args:
            archetypes: List of archetype dictionaries
            name: Archetype set name (without .json)
            description: Set description
            version: Set version
            demographic_context: Target demographic
            discovered: If True, save to discovered/ subdirectory
        
        Returns:
            Path to saved file
        """
        if discovered:
            filepath = self.archetypes_dir / "discovered" / f"{name}.json"
        else:
            filepath = self.archetypes_dir / f"{name}.json"
        
        data = {
            "name": name,
            "description": description,
            "version": version,
            "demographic_context": demographic_context,
            "archetypes": archetypes
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        cache_key = f"archetypes_{name}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        return filepath
    
    def get_archetype_sets(self, include_discovered: bool = False) -> List[str]:
        """
        Get list of available archetype sets.
        
        Args:
            include_discovered: Include discovered archetypes
        
        Returns:
            List of archetype set names
        """
        sets = []
        
        # Regular archetypes
        if self.archetypes_dir.exists():
            sets.extend([
                f.stem for f in self.archetypes_dir.glob("*.json")
            ])
        
        # Discovered archetypes
        if include_discovered:
            discovered_dir = self.archetypes_dir / "discovered"
            if discovered_dir.exists():
                sets.extend([
                    f"discovered/{f.stem}" for f in discovered_dir.glob("*.json")
                ])
        
        return sorted(sets)
    
    # ============================================================================
    # SYSTEM CONFIG
    # ============================================================================
    
    def load_system_config(self) -> Dict:
        """
        Load system configuration.
        
        Returns:
            System config dictionary
        """
        if "system_config" in self._cache:
            return self._cache["system_config"]
        
        filepath = self.config_dir / "system_config.json"
        
        if not filepath.exists():
            # Return defaults if no config exists
            return self._get_default_system_config()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._cache["system_config"] = data
        return data
    
    def save_system_config(self, config: Dict) -> Path:
        """
        Save system configuration.
        
        Args:
            config: System config dictionary
        
        Returns:
            Path to saved file
        """
        filepath = self.config_dir / "system_config.json"
        
        # Add metadata
        import datetime
        config['last_updated'] = datetime.datetime.now().isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        if "system_config" in self._cache:
            del self._cache["system_config"]
        
        return filepath
    
    def _get_default_system_config(self) -> Dict:
        """Get default system configuration"""
        return {
            "version": "1.0",
            "ollama": {
                "model": "gemma3:4b",
                "url": "http://localhost:11434",
                "temperature_persona": 0.7,
                "temperature_agent": 0.5,
                "timeout": 120,
                "max_retries": 3
            },
            "analysis": {
                "n_archetypes": 4,
                "target_r2": 0.80,
                "random_seed": 42,
                "max_archetypes_to_test": 8,
                "max_iterations": 5000
            },
            "simulation": {
                "n_calibration_samples": 10,
                "n_simulated_respondents": 1000,
                "stratified_sampling": False,
                "use_existing_personas": True
            },
            "data_generation": {
                "n_respondents": 200,
                "missing_rate": 0.0,
                "demographic_context": "University Students in New York"
            },
            "visualization": {
                "dpi": 300,
                "figure_format": "png",
                "style": "seaborn-v0_8-darkgrid",
                "color_palette": "husl",
                "n_colors": 8
            }
        }
    
    # ============================================================================
    # PERSONAS
    # ============================================================================
    
    def load_personas(self, name: str = "generated_personas") -> List[Dict]:
        """
        Load personas from JSON.
        
        Args:
            name: Persona set name (without .json)
        
        Returns:
            List of persona dictionaries
        """
        filepath = self.personas_dir / f"{name}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Personas not found at {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_personas(self, personas: List[Dict], name: str = "generated_personas") -> Path:
        """
        Save personas to JSON.
        
        Args:
            personas: List of persona dictionaries
            name: Persona set name (without .json)
        
        Returns:
            Path to saved file
        """
        filepath = self.personas_dir / f"{name}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(personas, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    
    def _validate_if_schema_exists(self, data: Dict, schema_filename: str):
        """Validate data against schema if schema exists"""
        schema_path = self.schema_dir / schema_filename
        
        if not schema_path.exists():
            return  # No schema, skip validation
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
    
    # ============================================================================
    # CACHE MANAGEMENT
    # ============================================================================
    
    def clear_cache(self):
        """Clear all cached configs"""
        self._cache.clear()
    
    def invalidate_cache(self, key: str):
        """Invalidate specific cache key"""
        if key in self._cache:
            del self._cache[key]


# Global instance
_loader = None

def get_loader() -> ConfigLoader:
    """Get global ConfigLoader instance"""
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader


# Convenience functions
def load_questions(template_name: str = "opinion_survey") -> List['Question']:
    """Load questions from template"""
    return get_loader().load_questions(template_name)


def save_questions(questions: List['Question'], name: str, **kwargs) -> Path:
    """Save questions to template"""
    return get_loader().save_questions(questions, name, **kwargs)


def load_archetypes(name: str = "default") -> List[Dict]:
    """Load archetype definitions"""
    return get_loader().load_archetypes(name)


def save_archetypes(archetypes: List[Dict], name: str, **kwargs) -> Path:
    """Save archetype definitions"""
    return get_loader().save_archetypes(archetypes, name, **kwargs)


def load_system_config() -> Dict:
    """Load system configuration"""
    return get_loader().load_system_config()


def save_system_config(config: Dict) -> Path:
    """Save system configuration"""
    return get_loader().save_system_config(config)


def load_personas(name: str = "generated_personas") -> List[Dict]:
    """Load personas"""
    return get_loader().load_personas(name)


def save_personas(personas: List[Dict], name: str = "generated_personas") -> Path:
    """Save personas"""
    return get_loader().save_personas(personas, name)


def get_question_templates() -> List[str]:
    """Get available question templates"""
    return get_loader().get_question_templates()


def get_archetype_sets(include_discovered: bool = False) -> List[str]:
    """Get available archetype sets"""
    return get_loader().get_archetype_sets(include_discovered)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING CONFIG LOADER")
    print("="*80 + "\n")
    
    loader = ConfigLoader()
    
    # Test system config
    print("Testing system config...")
    config = loader.load_system_config()
    print(f"✅ Loaded config with {len(config)} sections")
    
    # Test saving
    test_config = loader._get_default_system_config()
    test_config['test_field'] = 'test_value'
    saved_path = loader.save_system_config(test_config)
    print(f"✅ Saved config to: {saved_path}")
    
    # Test templates
    print("\nAvailable question templates:")
    templates = loader.get_question_templates()
    for t in templates:
        print(f"  • {t}")
    
    print("\nAvailable archetype sets:")
    sets = loader.get_archetype_sets(include_discovered=True)
    for s in sets:
        print(f"  • {s}")
    
    print("\n" + "="*80)
    print("✅ CONFIG LOADER TESTS PASSED")
    print("="*80 + "\n")