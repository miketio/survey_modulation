import React, { useState, useEffect } from 'react';
import { Database, Brain, Users, Play, Upload, RefreshCw, BarChart, AlertCircle, Star, Settings, CheckCircle, Download, Pause, FastForward, Edit2, Save, Trash2, PlusCircle, FileText, XCircle, Layers } from 'lucide-react';

const API_URL = 'http://localhost:8000/api';

export default function SurveyArchetypesApp() {
  const [activeTab, setActiveTab] = useState('archetypes');
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState(null);

  // Archetypes & Questions State
  const [initialArchetypes, setInitialArchetypes] = useState([]);
  const [questions, setQuestions] = useState([]);
  const [config, setConfig] = useState({
    demographicContext: 'University Students in New York',
    temperature: 0.7,
    randomSeed: 42,
    respondents: 200,
    calibrationSamples: 10,
    simulatedPopulation: 1000
  });


  const [availableQuestionTemplates, setAvailableQuestionTemplates] = useState([]);
  const [availableArchetypeSets, setAvailableArchetypeSets] = useState([]);
  const [templateQuestions, setTemplateQuestions] = useState({}); // Cache for loaded templates

  // Discovery State
  const [kResults, setKResults] = useState([]);
  const [selectedK, setSelectedK] = useState(null);
  const [analysisRunning, setAnalysisRunning] = useState(false);

  // Personas State
  const [personas, setPersonas] = useState([]);
  const [editingPersona, setEditingPersona] = useState(null);

  // Second Survey State
  const [secondSurveyQuestions, setSecondSurveyQuestions] = useState([]);

  // Calibration State
  const [calibrationInProgress, setCalibrationInProgress] = useState(false);
  const [calibrationData, setCalibrationData] = useState([]);
  const [calibrationProgress, setCalibrationProgress] = useState(0);

  // Simulation State
  const [simulationInProgress, setSimulationInProgress] = useState(false);
  const [simulatedData, setSimulatedData] = useState(null);
  const [simulationProgress, setSimulationProgress] = useState(0);

  // Visualization State
  const [visualizationUrl, setVisualizationUrl] = useState(null);

  // Load configurations from backend on mount
  useEffect(() => {
    const loadConfigurations = async () => {
      try {
        // 1. Load available question templates
        const templatesRes = await fetch(`${API_URL}/config/questions`);
        const templatesData = await templatesRes.json();
        setAvailableQuestionTemplates(templatesData.templates);
        
        // 2. Load default 'opinion_survey' if it exists
        if (templatesData.templates.includes('opinion_survey')) {
          const defaultRes = await fetch(`${API_URL}/config/questions/opinion_survey`);
          const defaultData = await defaultRes.json();
          
          setQuestions(defaultData.questions);
          
          // Cache it
          setTemplateQuestions(prev => ({
            ...prev,
            'opinion_survey': defaultData.questions
          }));
        }

        // 3. Load default 'validation_survey' for Tab 4 (Second Survey)
        if (templatesData.templates.includes('validation_survey')) {
          const validRes = await fetch(`${API_URL}/config/questions/validation_survey`);
          const validData = await validRes.json();
          
          setSecondSurveyQuestions(validData.questions);
          
          // Cache it
          setTemplateQuestions(prev => ({
            ...prev,
            'validation_survey': validData.questions
          }));
        }
        
        // 4. Load available archetype sets
        const archetypesRes = await fetch(`${API_URL}/config/archetypes`);
        const archetypesData = await archetypesRes.json();
        setAvailableArchetypeSets(archetypesData.archetype_sets);
        
        // 5. Load default archetypes
        if (archetypesData.archetype_sets.includes('default')) {
          const defaultArchRes = await fetch(`${API_URL}/config/archetypes/default`);
          const defaultArchData = await defaultArchRes.json();
          setInitialArchetypes(defaultArchData.archetypes);
        }
        
        // 6. Load system configuration
        const configRes = await fetch(`${API_URL}/config/system`);
        const configData = await configRes.json();
        setConfig({
          demographicContext: configData.data_generation.demographic_context,
          temperature: configData.ollama.temperature_persona,
          randomSeed: configData.analysis.random_seed,
          respondents: configData.data_generation.n_respondents,
          calibrationSamples: configData.simulation.n_calibration_samples,
          simulatedPopulation: configData.simulation.n_simulated_respondents
        });
        
        setStatusMessage('✅ Configurations loaded from backend');
        setTimeout(() => setStatusMessage(''), 2000);
        
      } catch (err) {
        console.error('Failed to load configurations:', err);
        setError('Failed to load configurations from server. Make sure backend is running.');
      }
    };
    
    loadConfigurations();
  }, []);

  // Helper function to load second survey template
  const handleLoadSecondSurveyTemplate = async (templateName) => {
    if (!templateName) return;
    
    try {
      // Check cache first
      if (templateQuestions[templateName]) {
        setSecondSurveyQuestions(templateQuestions[templateName]);
        setStatusMessage(`Loaded template: ${templateName}`);
        setTimeout(() => setStatusMessage(''), 2000);
        return;
      }
      
      // Fetch from backend
      const response = await fetch(`${API_URL}/config/questions/${templateName}`);
      if (!response.ok) throw new Error('Failed to load template');
      
      const data = await response.json();
      setSecondSurveyQuestions(data.questions);
      
      // Cache it
      setTemplateQuestions(prev => ({
        ...prev,
        [templateName]: data.questions
      }));
      
      setStatusMessage(`Loaded template: ${templateName}`);
      setTimeout(() => setStatusMessage(''), 2000);
    } catch (err) {
      setError(`Failed to load template: ${err.message}`);
    }
  };

  // === HANDLERS ===

  const handleGenerateData = async () => {
    setIsLoading(true);
    setStatusMessage('Generating synthetic survey data with your archetypes...');
    setError(null);

    try {
      const response = await fetch(`${API_URL}/data/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          questions: questions,
          config: config,
          archetypes: initialArchetypes
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate data');
      }

      const data = await response.json();
      setStatusMessage('Data generated! Moving to archetype discovery...');
      setTimeout(() => {
        setActiveTab('discovery');
        setStatusMessage('');
      }, 1500);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunKAnalysis = async () => {
    setAnalysisRunning(true);
    setIsLoading(true);
    setStatusMessage('Running archetypal analysis for k=2 to 8... (this may take a minute)');
    setError(null);

    try {
      const response = await fetch(`${API_URL}/analysis/k-comparison`, {
        method: 'POST'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data = await response.json();
      setKResults(data.results);
      setStatusMessage('Analysis complete! Select optimal k.');
      setTimeout(() => setStatusMessage(''), 3000);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
      setAnalysisRunning(false);
    }
  };

  const handleSelectK = async (k) => {
    setSelectedK(k);
    setIsLoading(true);
    setStatusMessage(`Generating personas for k=${k} using Ollama... (this may take 2-3 minutes)`);

    try {
      const response = await fetch(`${API_URL}/personas/generate?k=${k}`, {
        method: 'POST'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Persona generation failed');
      }

      const data = await response.json();
      setPersonas(data.personas);
      setStatusMessage('Personas generated! Review and edit if needed.');
      setTimeout(() => {
        setActiveTab('personas');
        setStatusMessage('');
      }, 1500);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSavePersona = async (updatedPersona) => {
    try {
      const response = await fetch(`${API_URL}/personas/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedPersona)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to update persona');
      }

      setPersonas(personas.map(p => 
        p.archetype_index === updatedPersona.archetype_index ? updatedPersona : p
      ));
      setEditingPersona(null);
      setStatusMessage('Persona saved!');
      setTimeout(() => setStatusMessage(''), 2000);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStartCalibration = async () => {
    setCalibrationInProgress(true);
    setCalibrationProgress(0);
    setCalibrationData([]);
    setStatusMessage('Starting calibration phase...');

    try {
      // ✅ Format questions properly
      const questionsPayload = secondSurveyQuestions.map(q => ({
        id: q.id,
        text: q.text,
        type: q.type,
        scale: q.scale,
        options: q.options
      }));

      console.log('📤 Sending to backend:', questionsPayload);

      // ✅ Send questions to backend
      const saveResponse = await fetch(`${API_URL}/survey/questions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(questionsPayload)
      });

      if (!saveResponse.ok) {
        const errorData = await saveResponse.json().catch(() => ({}));
        console.error('❌ Backend error:', errorData);
        throw new Error(errorData.detail || 'Failed to save questions to backend');
      }

      const saveResult = await saveResponse.json();
      console.log('✅ Backend saved:', saveResult);

      const ws = new WebSocket('ws://localhost:8000/api/calibration/live');
      
      const timeout = setTimeout(() => {
        ws.close();
        setError('Calibration timed out after 5 minutes');
        setCalibrationInProgress(false);
      }, 1000000);
      
      ws.onopen = () => {
        console.log('🔌 WebSocket connected');
        ws.send(JSON.stringify({
          n_samples: config.calibrationSamples
        }));
        setStatusMessage(`Calibrating: Each agent answering ${config.calibrationSamples} times per question...`);
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        console.log('📨 Received:', message.type);
        
        if (message.type === 'calibration_update') {
          setCalibrationData(prev => [...prev, message.data]);
          setCalibrationProgress(message.progress);
        } else if (message.type === 'calibration_complete') {
          clearTimeout(timeout);
          setCalibrationInProgress(false);
          setStatusMessage('Calibration complete! Ready to simulate population.');
          ws.close();
          setTimeout(() => {
            setActiveTab('simulation');
            setStatusMessage('');
          }, 2000);
        } else if (message.type === 'error') {
          clearTimeout(timeout);
          setError(message.message);
          setCalibrationInProgress(false);
          ws.close();
        }
      };

      ws.onerror = (error) => {
        clearTimeout(timeout);
        console.error('❌ WebSocket error:', error);
        setError('Connection error. Make sure backend is running.');
        setCalibrationInProgress(false);
      };
      
      ws.onclose = () => {
        clearTimeout(timeout);
        console.log('🔌 WebSocket closed');
      };

    } catch (err) {
      console.error('❌ Calibration error:', err);
      setError(err.message);
      setCalibrationInProgress(false);
    }
  };

  const handleStartSimulation = async () => {
    setSimulationInProgress(true);
    setSimulationProgress(0);
    setStatusMessage(`Simulating ${config.simulatedPopulation} respondents...`);

    try {
      const response = await fetch(`${API_URL}/simulation/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_respondents: config.simulatedPopulation
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Simulation failed');
      }

      const data = await response.json();
      setSimulatedData(data);
      setSimulationInProgress(false);
      setStatusMessage('Simulation complete! View results and visualizations.');
      setTimeout(() => {
        setActiveTab('analysis');
        setStatusMessage('');
      }, 2000);

    } catch (err) {
      setError(err.message);
      setSimulationInProgress(false);
    }
  };

  const handleGenerateVisualization = async () => {
    setIsLoading(true);
    setStatusMessage('Generating visualization...');

    try {
      const response = await fetch(`${API_URL}/visualization/simulated-population`, {
        method: 'POST'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Visualization failed');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setVisualizationUrl(url);
      setStatusMessage('Visualization created!');
      setTimeout(() => setStatusMessage(''), 2000);

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddQuestion = () => {
    const newId = `Q${questions.length + 1}`;
    setQuestions([...questions, {
      id: newId,
      text: 'New question',
      type: 'likert',
      scale: '1-5'
    }]);
  };

  const handleDeleteQuestion = (index) => {
    setQuestions(questions.filter((_, i) => i !== index));
  };

  const handleAddArchetype = () => {
    setInitialArchetypes([...initialArchetypes, {
      name: `New Archetype ${initialArchetypes.length + 1}`,
      opinion_pattern: [3, 3, 3, 3, 3],
      weight: 0.10
    }]);
  };

  const handleDeleteArchetype = (index) => {
    setInitialArchetypes(initialArchetypes.filter((_, i) => i !== index));
  };

  const normalizeWeights = () => {
    const total = initialArchetypes.reduce((sum, a) => sum + a.weight, 0);
    if (total > 0) {
      setInitialArchetypes(initialArchetypes.map(a => ({
        ...a,
        weight: a.weight / total
      })));
    }
  };

  const handleLoadTemplate = async (templateName) => {
    if (!templateName) return;
    
    try {
      const response = await fetch(`${API_URL}/config/questions/${templateName}`);
      if (!response.ok) throw new Error('Failed to load template');
      
      const data = await response.json();
      setQuestions(data.questions);
      setStatusMessage(`Loaded template: ${templateName}`);
      setTimeout(() => setStatusMessage(''), 2000);
    } catch (err) {
      setError(`Failed to load template: ${err.message}`);
    }
  };

  const handleSaveTemplate = async (templateName) => {
    if (!templateName) {
      setError('Please provide a template name');
      return;
    }
    
    try {
      const response = await fetch(`${API_URL}/config/questions/${templateName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          questions: questions,
          description: `Custom template: ${templateName}`,
          version: '1.0'
        })
      });
      
      if (!response.ok) throw new Error('Failed to save template');
      
      setStatusMessage(`Template '${templateName}' saved successfully!`);
      
      // Reload available templates
      const templatesRes = await fetch(`${API_URL}/config/questions`);
      const templatesData = await templatesRes.json();
      setAvailableQuestionTemplates(templatesData.templates);
      
      setTimeout(() => setStatusMessage(''), 2000);
    } catch (err) {
      setError(`Failed to save template: ${err.message}`);
    }
  };
  const handleDownload = async (endpoint, filename) => {
    try {
      const response = await fetch(`${API_URL}${endpoint}`);
      if (!response.ok) throw new Error('Download failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      setStatusMessage(`Downloaded ${filename}`);
      setTimeout(() => setStatusMessage(''), 2000);
    } catch (err) {
      setError(err.message);
    }
  };

  // === COMPONENTS ===

  const NavButton = ({ active, onClick, icon, label, disabled }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all
        ${active 
          ? 'bg-slate-800 text-white shadow-sm' 
          : 'text-slate-300 hover:bg-slate-800 hover:text-white'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      {icon}
      {label}
    </button>
  );

  // Archetype Row Component - isolated state management
  const ArchetypeRow = React.memo(({ archetype, index, onUpdate, onDelete, canDelete }) => {
    const [localArchetype, setLocalArchetype] = useState(archetype);

    useEffect(() => {
      setLocalArchetype(archetype);
    }, [archetype]);

    const handleBlur = () => {
      onUpdate(index, localArchetype);
    };

    const handleLocalChange = (field, value) => {
      setLocalArchetype(prev => ({...prev, [field]: value}));
    };

    const handlePatternChange = (patternIdx, value) => {
      setLocalArchetype(prev => {
        const newPattern = [...prev.opinion_pattern];
        newPattern[patternIdx] = parseFloat(value) || 1;
        return {...prev, opinion_pattern: newPattern};
      });
    };

    return (
      <tr className="border-t">
        <td className="px-4 py-2">
          <input
            type="text"
            value={localArchetype.name}
            onChange={(e) => handleLocalChange('name', e.target.value)}
            onBlur={handleBlur}
            className="w-full px-2 py-1 border rounded"
          />
        </td>
        <td className="px-4 py-2">
          <div className="flex gap-2">
            {localArchetype.opinion_pattern.map((val, qIdx) => (
              <input
                key={qIdx}
                type="number"
                min="1"
                max="5"
                step="0.1"
                value={val}
                onChange={(e) => handlePatternChange(qIdx, e.target.value)}
                onBlur={handleBlur}
                className="w-12 px-2 py-1 border rounded text-center"
              />
            ))}
          </div>
        </td>
        <td className="px-4 py-2">
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={localArchetype.weight}
            onChange={(e) => handleLocalChange('weight', parseFloat(e.target.value) || 0)}
            onBlur={handleBlur}
            className="w-full px-2 py-1 border rounded"
          />
        </td>
        <td className="px-4 py-2">
          <button 
            onClick={() => onDelete(index)}
            className="text-red-600 hover:text-red-800"
            disabled={!canDelete}
          >
            <Trash2 size={16} />
          </button>
        </td>
      </tr>
    );
  });

  const ArchetypesEditor = () => {
    const handleArchetypeUpdate = (index, updatedArchetype) => {
      setInitialArchetypes(prev => {
        const newArchetypes = [...prev];
        newArchetypes[index] = updatedArchetype;
        return newArchetypes;
      });
    };

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-lg font-bold">Initial Archetypes</h3>
            <p className="text-sm text-slate-600">Define the "true" archetypes that will generate synthetic survey data</p>
          </div>
          <div className="flex gap-2">
            <button 
              onClick={handleAddArchetype}
              className="px-3 py-2 bg-blue-600 text-white rounded-md text-sm flex items-center gap-2 hover:bg-blue-700"
            >
              <PlusCircle size={16} /> Add Archetype
            </button>
            <button 
              onClick={normalizeWeights}
              className="px-3 py-2 bg-green-600 text-white rounded-md text-sm flex items-center gap-2 hover:bg-green-700"
            >
              <RefreshCw size={16} /> Normalize Weights
            </button>
          </div>
        </div>

        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-100">
              <tr>
                <th className="px-4 py-2 text-left">Name</th>
                <th className="px-4 py-2 text-left">Opinion Pattern (Q1-Q5: 1-5 scale)</th>
                <th className="px-4 py-2 text-left w-24">Weight</th>
                <th className="px-4 py-2 w-16"></th>
              </tr>
            </thead>
            <tbody>
              {initialArchetypes.map((archetype, idx) => (
                <ArchetypeRow
                  key={`arch-${idx}`}
                  archetype={archetype}
                  index={idx}
                  onUpdate={handleArchetypeUpdate}
                  onDelete={handleDeleteArchetype}
                  canDelete={initialArchetypes.length > 2}
                />
              ))}
            </tbody>
          </table>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle size={16} className="text-blue-600" />
            <span className="text-sm font-medium text-blue-900">Weight Sum: {initialArchetypes.reduce((sum, a) => sum + a.weight, 0).toFixed(2)}</span>
          </div>
          <p className="text-xs text-blue-700">Weights should sum to 1.0. Click "Normalize Weights" to auto-adjust.</p>
        </div>
      </div>
    );
  };

  // Question Row Component - isolated state management
  const QuestionRow = React.memo(({ question, index, onUpdate, onDelete }) => {
    const [localQuestion, setLocalQuestion] = useState(question);

    useEffect(() => {
      setLocalQuestion(question);
    }, [question]);

    const handleBlur = () => {
      onUpdate(index, localQuestion);
    };

    const handleLocalChange = (field, value) => {
      setLocalQuestion(prev => ({...prev, [field]: value}));
    };

    const handleTypeChange = (newType) => {
      const newQuestion = { ...localQuestion, type: newType };
      if (newType === 'likert') {
        newQuestion.scale = '1-5';
        delete newQuestion.options;
      } else {
        delete newQuestion.scale;
        newQuestion.options = 'Option1,Option2,Option3';
      }
      setLocalQuestion(newQuestion);
      onUpdate(index, newQuestion);
    };

    return (
      <tr className="border-t">
        <td className="px-4 py-2">
          <input
            type="text"
            value={localQuestion.id}
            onChange={(e) => handleLocalChange('id', e.target.value)}
            onBlur={handleBlur}
            className="w-full px-2 py-1 border rounded"
          />
        </td>
        <td className="px-4 py-2">
          <input
            type="text"
            value={localQuestion.text}
            onChange={(e) => handleLocalChange('text', e.target.value)}
            onBlur={handleBlur}
            className="w-full px-2 py-1 border rounded"
          />
        </td>
        <td className="px-4 py-2">
          <select
            value={localQuestion.type}
            onChange={(e) => handleTypeChange(e.target.value)}
            className="w-full px-2 py-1 border rounded"
          >
            <option value="likert">Likert</option>
            <option value="categorical">Categorical</option>
            <option value="ordinal">Ordinal</option>
          </select>
        </td>
        <td className="px-4 py-2">
          {localQuestion.type === 'likert' ? (
            <input
              type="text"
              value={localQuestion.scale || '1-5'}
              onChange={(e) => handleLocalChange('scale', e.target.value)}
              onBlur={handleBlur}
              className="w-full px-2 py-1 border rounded text-xs"
              placeholder="1-5"
            />
          ) : (
            <input
              type="text"
              value={localQuestion.options || ''}
              onChange={(e) => handleLocalChange('options', e.target.value)}
              onBlur={handleBlur}
              className="w-full px-2 py-1 border rounded text-xs"
              placeholder="Option1,Option2,Option3"
            />
          )}
        </td>
        <td className="px-4 py-2">
          <button 
            onClick={() => onDelete(index)}
            className="text-red-600 hover:text-red-800"
          >
            <Trash2 size={16} />
          </button>
        </td>
      </tr>
    );
  });

  const QuestionEditor = () => {
    const handleQuestionUpdate = (index, updatedQuestion) => {
      setQuestions(prev => {
        const newQuestions = [...prev];
        newQuestions[index] = updatedQuestion;
        return newQuestions;
      });
    };

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-bold">Survey Questions</h3>
          <div className="flex gap-2">
            <select 
              className="px-3 py-2 border rounded-md text-sm"
              onChange={(e) => handleLoadTemplate(e.target.value)}
            >
              <option value="">Load Template...</option>
              {availableQuestionTemplates.map(template => (
                <option key={template} value={template}>
                  {template.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
            <button 
              onClick={() => {
                const name = prompt('Enter template name:');
                if (name) handleSaveTemplate(name);
              }}
              className="px-3 py-2 bg-green-600 text-white rounded-md text-sm flex items-center gap-2 hover:bg-green-700"
            >
              <Save size={16} /> Save as Template
            </button>
            <button 
              onClick={handleAddQuestion}
              className="px-3 py-2 bg-blue-600 text-white rounded-md text-sm flex items-center gap-2 hover:bg-blue-700"
            >
              <PlusCircle size={16} /> Add Question
            </button>

          </div>
        </div>

        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-100">
              <tr>
                <th className="px-4 py-2 text-left w-20">ID</th>
                <th className="px-4 py-2 text-left">Question Text</th>
                <th className="px-4 py-2 text-left w-32">Type</th>
                <th className="px-4 py-2 text-left w-48">Scale/Options</th>
                <th className="px-4 py-2 w-16"></th>
              </tr>
            </thead>
            <tbody>
              {questions.map((q, idx) => (
                <QuestionRow
                  key={`q-${idx}`}
                  question={q}
                  index={idx}
                  onUpdate={handleQuestionUpdate}
                  onDelete={handleDeleteQuestion}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const ConfigPanel = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-bold">Configuration</h3>
      
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Demographic Context</label>
          <input 
            type="text" 
            value={config.demographicContext}
            onChange={(e) => setConfig({...config, demographicContext: e.target.value})}
            className="w-full px-3 py-2 border rounded-md"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Initial Survey Respondents</label>
          <input 
            type="number" 
            value={config.respondents}
            onChange={(e) => setConfig({...config, respondents: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border rounded-md"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Calibration Samples per Agent</label>
          <input 
            type="number" 
            value={config.calibrationSamples}
            onChange={(e) => setConfig({...config, calibrationSamples: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-slate-500 mt-1">How many times each agent answers each question</p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Simulated Population Size</label>
          <input 
            type="number" 
            value={config.simulatedPopulation}
            onChange={(e) => setConfig({...config, simulatedPopulation: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-slate-500 mt-1">Final synthetic dataset size</p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Temperature (LLM)</label>
          <input 
            type="number" 
            step="0.1"
            min="0"
            max="2"
            value={config.temperature}
            onChange={(e) => setConfig({...config, temperature: parseFloat(e.target.value)})}
            className="w-full px-3 py-2 border rounded-md"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Random Seed</label>
          <input 
            type="number" 
            value={config.randomSeed}
            onChange={(e) => setConfig({...config, randomSeed: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border rounded-md"
          />
        </div>
      </div>
    </div>
  );

  const CalibrationTable = () => {
    if (calibrationData.length === 0) return null;

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-bold">Calibration Results</h3>
          <div className="text-sm text-slate-600">
            Progress: {calibrationProgress.toFixed(0)}%
          </div>
        </div>

        <div className="border rounded-lg overflow-hidden">
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-sm">
              <thead className="bg-slate-100 sticky top-0">
                <tr>
                  <th className="px-4 py-2 text-left">Persona</th>
                  <th className="px-4 py-2 text-left">Question</th>
                  <th className="px-4 py-2 text-center">Modal</th>
                  <th className="px-4 py-2 text-center">Mean</th>
                  <th className="px-4 py-2 text-center">Std</th>
                  <th className="px-4 py-2 text-center">P(↓)</th>
                  <th className="px-4 py-2 text-center">P(=)</th>
                  <th className="px-4 py-2 text-center">P(↑)</th>
                </tr>
              </thead>
              <tbody>
                {calibrationData.map((row, idx) => (
                  <tr key={idx} className="border-t hover:bg-slate-50">
                    <td className="px-4 py-2 font-medium">{row.persona_name}</td>
                    <td className="px-4 py-2 text-xs">{row.question_id}</td>
                    <td className="px-4 py-2 text-center font-bold text-blue-600">{row.modal_answer}</td>
                    <td className="px-4 py-2 text-center">{row.mean_answer?.toFixed(2)}</td>
                    <td className="px-4 py-2 text-center text-slate-600">{row.std?.toFixed(2)}</td>
                    <td className="px-4 py-2 text-center text-red-600">{(row.p_lower * 100).toFixed(0)}%</td>
                    <td className="px-4 py-2 text-center text-slate-600">{(row.p_stay * 100).toFixed(0)}%</td>
                    <td className="px-4 py-2 text-center text-green-600">{(row.p_higher * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  const SimulationSummary = () => {
    if (!simulatedData) return null;

    return (
      <div className="space-y-4">
        <h3 className="text-lg font-bold">Simulation Summary</h3>
        
        <div className="grid md:grid-cols-4 gap-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="text-sm text-blue-600 font-medium mb-1">Total Respondents</div>
            <div className="text-2xl font-bold text-blue-900">{simulatedData.total_respondents}</div>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="text-sm text-green-600 font-medium mb-1">Archetypes</div>
            <div className="text-2xl font-bold text-green-900">{simulatedData.n_archetypes}</div>
          </div>
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <div className="text-sm text-purple-600 font-medium mb-1">Questions</div>
            <div className="text-2xl font-bold text-purple-900">{simulatedData.n_questions}</div>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <div className="text-sm text-amber-600 font-medium mb-1">Answers</div>
            <div className="text-2xl font-bold text-amber-900">{simulatedData.total_answers}</div>
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <h4 className="font-bold mb-2">Archetype Distribution</h4>
          <div className="space-y-2">
            {Object.entries(simulatedData.archetype_distribution || {}).map(([name, count]) => (
              <div key={name} className="flex items-center gap-2">
                <div className="w-32 text-sm truncate">{name}</div>
                <div className="flex-1 bg-slate-100 rounded-full h-6 overflow-hidden">
                  <div 
                    className="bg-blue-500 h-full flex items-center justify-center text-white text-xs font-bold"
                    style={{ width: `${(count / simulatedData.total_respondents) * 100}%` }}
                  >
                    {count}
                  </div>
                </div>
                <div className="w-16 text-sm text-right">{((count / simulatedData.total_respondents) * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const PersonaCard = ({ persona, index }) => {
    const isEditing = editingPersona?.archetype_index === persona.archetype_index;
    const [editData, setEditData] = useState(persona);
    
    // Local string state for typing freedom
    const [valuesText, setValuesText] = useState('');
    const [fearsText, setFearsText] = useState('');

    useEffect(() => {
      if (isEditing) {
        setEditData(persona);
        // Convert arrays to comma-separated strings for editing
        setValuesText(Array.isArray(persona.values) ? persona.values.join(', ') : '');
        setFearsText(Array.isArray(persona.fears) ? persona.fears.join(', ') : '');
      }
    }, [isEditing, persona]);

    const handleSave = () => {
      // Parse comma-separated text into arrays only on save
      const parsedValues = valuesText
        .split(',')
        .map(v => v.trim())
        .filter(v => v.length > 0);
      
      const parsedFears = fearsText
        .split(',')
        .map(f => f.trim())
        .filter(f => f.length > 0);

      const finalData = {
        ...editData,
        values: parsedValues,
        fears: parsedFears
      };

      handleSavePersona(finalData);
    };

    if (isEditing) {
      return (
        <div className="bg-white border-2 border-blue-500 rounded-xl p-6 space-y-4 shadow-lg">
          <div className="flex justify-between items-start">
            <h3 className="text-lg font-bold">Editing Persona {index + 1}</h3>
          </div>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium mb-1">Name (with age)</label>
              <input
                type="text"
                value={editData.name || ''}
                onChange={(e) => setEditData({...editData, name: e.target.value})}
                className="w-full px-3 py-2 border rounded-md text-sm"
                placeholder="e.g., Sarah, 22"
              />
            </div>

            <div>
              <label className="block text-xs font-medium mb-1">Occupation</label>
              <input
                type="text"
                value={editData.occupation || ''}
                onChange={(e) => setEditData({...editData, occupation: e.target.value})}
                className="w-full px-3 py-2 border rounded-md text-sm"
              />
            </div>

            <div>
              <label className="block text-xs font-medium mb-1">Worldview (2-3 sentences)</label>
              <textarea
                value={editData.worldview || ''}
                onChange={(e) => setEditData({...editData, worldview: e.target.value})}
                rows={3}
                className="w-full px-3 py-2 border rounded-md text-sm"
              />
            </div>

            <div>
              <label className="block text-xs font-medium mb-1">Values</label>
              <input
                type="text"
                value={valuesText}
                onChange={(e) => setValuesText(e.target.value)}
                className="w-full px-3 py-2 border rounded-md text-sm"
                placeholder="Innovation, Work-life balance, Community engagement, etc."
              />
              <p className="text-xs text-slate-500 mt-1">Separate with commas. Type freely—parsed on save.</p>
            </div>

            <div>
              <label className="block text-xs font-medium mb-1">Fears</label>
              <input
                type="text"
                value={fearsText}
                onChange={(e) => setFearsText(e.target.value)}
                className="w-full px-3 py-2 border rounded-md text-sm"
                placeholder="Unemployment, Climate change, Economic instability, etc."
              />
              <p className="text-xs text-slate-500 mt-1">Separate with commas. Type freely—parsed on save.</p>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleSave}
              className="flex-1 bg-green-600 hover:bg-green-700 text-white py-2 rounded-md font-medium flex items-center justify-center gap-2"
            >
              <Save size={16} /> Save Changes
            </button>
            <button
              onClick={() => setEditingPersona(null)}
              className="px-4 bg-slate-200 hover:bg-slate-300 text-slate-700 py-2 rounded-md font-medium"
            >
              Cancel
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
        <div className={`h-2 ${index % 3 === 0 ? 'bg-blue-500' : index % 3 === 1 ? 'bg-purple-500' : 'bg-emerald-500'}`} />
        
        <div className="p-6 space-y-4">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-xl font-bold text-slate-900">{persona.name}</h3>
              <p className="text-sm text-slate-600 flex items-center gap-1 mt-1">
                <Users size={14} /> {persona.occupation}
              </p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <span className="bg-slate-100 px-3 py-1 rounded-full text-xs font-bold border border-slate-200">
                {(persona.weight * 100).toFixed(0)}% Prevalence
              </span>
              <button
                onClick={() => setEditingPersona(persona)}
                className="text-blue-600 hover:text-blue-800 flex items-center gap-1 text-sm font-medium"
              >
                <Edit2 size={14} /> Edit
              </button>
            </div>
          </div>

          <div className="relative">
            <div className="absolute -left-2 top-0 text-3xl text-slate-200 font-serif">"</div>
            <p className="pl-4 italic text-slate-600 text-sm leading-relaxed">
              {persona.worldview}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4 pt-4 border-t">
            <div>
              <h4 className="text-xs font-bold uppercase text-blue-600 mb-2 flex items-center gap-1">
                <Star size={12} fill="currentColor" /> Values
              </h4>
              <div className="flex flex-wrap gap-1">
                {persona.values?.map((v, i) => (
                  <span key={i} className="px-2 py-1 bg-blue-50 text-blue-700 rounded-md text-xs font-medium border border-blue-100">
                    {v}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-xs font-bold uppercase text-rose-600 mb-2 flex items-center gap-1">
                <AlertCircle size={12} fill="currentColor" /> Fears
              </h4>
              <div className="flex flex-wrap gap-1">
                {persona.fears?.map((f, i) => (
                  <span key={i} className="px-2 py-1 bg-rose-50 text-rose-700 rounded-md text-xs font-medium border border-rose-100">
                    {f}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // === MAIN RENDER ===

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-800 flex flex-col">
      {/* Navigation */}
      <nav className="bg-slate-900 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-2">
              <Brain className="w-6 h-6 text-blue-400" />
              <span className="font-bold text-lg">Survey Archetypes</span>
            </div>
            
            <div className="flex space-x-1">
              <NavButton active={activeTab === 'archetypes'} onClick={() => setActiveTab('archetypes')} icon={<Layers size={16} />} label="0. Archetypes" />
              <NavButton active={activeTab === 'setup'} onClick={() => setActiveTab('setup')} icon={<Settings size={16} />} label="1. Setup" />
              <NavButton active={activeTab === 'discovery'} onClick={() => setActiveTab('discovery')} icon={<BarChart size={16} />} label="2. Discovery" disabled={questions.length === 0} />
              <NavButton active={activeTab === 'personas'} onClick={() => setActiveTab('personas')} icon={<Users size={16} />} label="3. Personas" disabled={personas.length === 0} />
              <NavButton active={activeTab === 'survey'} onClick={() => setActiveTab('survey')} icon={<FileText size={16} />} label="4. Survey" disabled={personas.length === 0} />
              <NavButton active={activeTab === 'calibration'} onClick={() => setActiveTab('calibration')} icon={<RefreshCw size={16} />} label="5. Calibration" disabled={secondSurveyQuestions.length === 0} />
              <NavButton active={activeTab === 'simulation'} onClick={() => setActiveTab('simulation')} icon={<Play size={16} />} label="6. Simulation" disabled={calibrationData.length === 0} />
              <NavButton active={activeTab === 'analysis'} onClick={() => setActiveTab('analysis')} icon={<Download size={16} />} label="7. Analysis" disabled={!simulatedData} />
            </div>
          </div>
        </div>
      </nav>

      {/* Status Messages */}
      {isLoading && (
        <div className="fixed bottom-6 right-6 bg-blue-600 text-white px-6 py-3 rounded-full shadow-xl flex items-center space-x-3 z-50 animate-pulse">
          <RefreshCw className="w-5 h-5 animate-spin" />
          <span className="font-medium">{statusMessage}</span>
        </div>
      )}

      {error && (
        <div className="max-w-7xl mx-auto w-full px-6 mt-6">
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r shadow-sm flex items-center justify-between">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 text-red-500 mr-3" />
              <span className="text-red-700 font-medium">{error}</span>
            </div>
            <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700">
              <XCircle size={20} />
            </button>
          </div>
        </div>
      )}

      {statusMessage && !isLoading && (
        <div className="max-w-7xl mx-auto w-full px-6 mt-6">
          <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded-r shadow-sm flex items-center justify-between">
            <div className="flex items-center">
              <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
              <span className="text-green-700 font-medium">{statusMessage}</span>
            </div>
            <button onClick={() => setStatusMessage('')} className="text-green-500 hover:text-green-700">
              <XCircle size={20} />
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto w-full p-6">
        
        {/* Tab 0: Initial Archetypes */}
        {activeTab === 'archetypes' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <ArchetypesEditor />
            </div>

            <button
              onClick={() => setActiveTab('setup')}
              disabled={initialArchetypes.length < 2}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed text-white py-4 rounded-lg font-bold text-lg shadow-lg transition-all"
            >
              Continue to Setup →
            </button>
          </div>
        )}

        {/* Tab 1: Setup */}
        {activeTab === 'setup' && (
          <div className="space-y-6">
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-amber-900">Using {initialArchetypes.length} initial archetypes</p>
                <button 
                  onClick={() => setActiveTab('archetypes')}
                  className="text-sm text-amber-700 underline hover:text-amber-900 mt-1"
                >
                  Click here to change initial archetypes
                </button>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border p-6">
              <QuestionEditor />
            </div>

            <div className="bg-white rounded-xl shadow-sm border p-6">
              <ConfigPanel />
            </div>

            <button
              onClick={handleGenerateData}
              disabled={isLoading || questions.length === 0}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed text-white py-4 rounded-lg font-bold text-lg shadow-lg flex items-center justify-center gap-2 transition-all"
            >
              <Database size={20} />
              Generate Initial Data
            </button>
          </div>
        )}

        {/* Tab 2: Discovery - K Analysis (unchanged) */}
        {activeTab === 'discovery' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-bold">k-vs-R² Analysis</h3>
                {kResults.length === 0 && (
                  <button
                    onClick={handleRunKAnalysis}
                    disabled={analysisRunning}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium flex items-center gap-2"
                  >
                    <Play size={16} /> Run Analysis
                  </button>
                )}
              </div>

              {kResults.length > 0 && (
                <>
                  <div className="border rounded-lg overflow-hidden mb-4">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-100">
                        <tr>
                          <th className="px-4 py-3 text-left w-16">k</th>
                          <th className="px-4 py-3 text-left w-24">R²</th>
                          <th className="px-4 py-3 text-left">Distribution</th>
                          <th className="px-4 py-3 w-12"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {kResults.map((result) => (
                          <tr 
                            key={result.k} 
                            className={`border-t cursor-pointer hover:bg-slate-50 ${selectedK === result.k ? 'bg-blue-50 border-l-4 border-l-blue-600' : ''}`}
                            onClick={() => setSelectedK(result.k)}
                          >
                            <td className="px-4 py-3 font-bold">{result.k}</td>
                            <td className="px-4 py-3">
                              <span className={`font-medium ${result.r2 >= 0.8 ? 'text-green-600' : result.r2 >= 0.65 ? 'text-amber-600' : 'text-red-600'}`}>
                                {(result.r2 * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex gap-1">
                                {result.proportions.map((p, idx) => (
                                  <div 
                                    key={idx}
                                    style={{ width: `${p * 100}%` }}
                                    className={`h-8 rounded text-xs flex items-center justify-center text-white font-medium ${
                                      idx % 3 === 0 ? 'bg-blue-500' : idx % 3 === 1 ? 'bg-purple-500' : 'bg-emerald-500'
                                    }`}
                                  >
                                    {(p * 100).toFixed(0)}%
                                  </div>
                                ))}
                              </div>
                            </td>
                            <td className="px-4 py-3 text-center">
                              {selectedK === result.k && (
                                <CheckCircle size={20} className="text-blue-600 inline-block" />
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {selectedK && (
                    <button
                      onClick={() => handleSelectK(selectedK)}
                      disabled={isLoading}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-bold text-lg shadow-lg"
                    >
                      Generate Personas for k={selectedK}
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Tab 3: Personas (unchanged) */}
        {activeTab === 'personas' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center bg-white p-6 rounded-xl shadow-sm border">
              <div>
                <h2 className="text-2xl font-bold text-slate-800">Generated Personas</h2>
                <p className="text-slate-500">Review and edit personas. Changes override LLM generation.</p>
              </div>
              <div className="text-sm text-slate-600 bg-slate-100 px-4 py-2 rounded-lg">
                <span className="font-bold">{personas.length}</span> archetypes selected
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {personas.map((p, idx) => (
                <PersonaCard key={p.archetype_index} persona={p} index={idx} />
              ))}
            </div>

            <button
              onClick={() => setActiveTab('survey')}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-4 rounded-lg font-bold text-lg shadow-lg"
            >
              Continue to Second Survey →
            </button>
          </div>
        )}

        {/* Tab 4: Second Survey Questions */}
        {activeTab === 'survey' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="text-lg font-bold mb-4">Second Survey Questions</h3>
              <p className="text-sm text-slate-600 mb-4">Define validation questions that AI agents will answer. Supports Likert, Categorical, and Ordinal questions.</p>
              
              <div className="flex gap-2 mb-4">
                <select 
                  className="px-3 py-2 border rounded-md text-sm flex-grow"
                  onChange={(e) => handleLoadSecondSurveyTemplate(e.target.value)}
                  defaultValue=""
                >
                  <option value="" disabled>Load Template...</option>
                  {availableQuestionTemplates.map(template => (
                    <option key={template} value={template}>
                      {template.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </option>
                  ))}
                </select>

                <button 
                  onClick={() => setSecondSurveyQuestions([...secondSurveyQuestions, {
                    id: `S${secondSurveyQuestions.length + 1}`,
                    text: 'New question',
                    type: 'likert',
                    scale: '1-5'
                  }])}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm flex items-center gap-2 transition-colors shadow-sm"
                >
                  <PlusCircle size={16} /> Add
                </button>
              </div>


              <div className="border rounded-lg overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-100">
                    <tr>
                      <th className="px-4 py-2 text-left w-20">ID</th>
                      <th className="px-4 py-2 text-left">Question</th>
                      <th className="px-4 py-2 text-left w-32">Type</th>
                      <th className="px-4 py-2 text-left w-48">Scale/Options</th>
                      <th className="px-4 py-2 w-16"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {secondSurveyQuestions.map((q, idx) => (
                      <tr key={idx} className="border-t">
                        <td className="px-4 py-2 font-mono text-xs">{q.id}</td>
                        <td className="px-4 py-2">
                          <input
                            type="text"
                            value={q.text}
                            onChange={(e) => {
                              const newQuestions = [...secondSurveyQuestions];
                              newQuestions[idx].text = e.target.value;
                              setSecondSurveyQuestions(newQuestions);
                            }}
                            className="w-full px-2 py-1 border rounded"
                          />
                        </td>
                        <td className="px-4 py-2">
                          <select
                            value={q.type}
                            onChange={(e) => {
                              const newType = e.target.value;
                              const newQuestion = { ...q, type: newType };
                              if (newType === 'likert') {
                                newQuestion.scale = '1-5';
                                delete newQuestion.options;
                              } else {
                                delete newQuestion.scale;
                                newQuestion.options = 'Option1,Option2,Option3';
                              }
                              const newQuestions = [...secondSurveyQuestions];
                              newQuestions[idx] = newQuestion;
                              setSecondSurveyQuestions(newQuestions);
                            }}
                            className="w-full px-2 py-1 border rounded text-xs"
                          >
                            <option value="likert">Likert</option>
                            <option value="categorical">Categorical</option>
                            <option value="ordinal">Ordinal</option>
                          </select>
                        </td>
                        <td className="px-4 py-2">
                          {q.type === 'likert' ? (
                            <input
                              type="text"
                              value={q.scale || '1-5'}
                              onChange={(e) => {
                                const newQuestions = [...secondSurveyQuestions];
                                newQuestions[idx].scale = e.target.value;
                                setSecondSurveyQuestions(newQuestions);
                              }}
                              className="w-full px-2 py-1 border rounded text-xs"
                              placeholder="1-5"
                            />
                          ) : (
                            <input
                              type="text"
                              value={q.options || ''}
                              onChange={(e) => {
                                const newQuestions = [...secondSurveyQuestions];
                                newQuestions[idx].options = e.target.value;
                                setSecondSurveyQuestions(newQuestions);
                              }}
                              className="w-full px-2 py-1 border rounded text-xs"
                              placeholder="Option1,Option2,Option3"
                            />
                          )}
                        </td>
                        <td className="px-4 py-2">
                          <button 
                            onClick={() => setSecondSurveyQuestions(secondSurveyQuestions.filter((_, i) => i !== idx))}
                            className="text-red-600 hover:text-red-800"
                          >
                            <Trash2 size={16} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-xs text-blue-700">
                  <strong>Question Types:</strong><br/>
                  • <strong>Likert:</strong> 1-5 scale (e.g., strongly disagree to strongly agree)<br/>
                  • <strong>Categorical:</strong> Multiple choice with no inherent order (e.g., employment status)<br/>
                  • <strong>Ordinal:</strong> Ordered categories (e.g., income levels: low, medium, high)
                </p>
              </div>
            </div>

            <button
              onClick={() => {
                console.log('📋 Questions to send:', secondSurveyQuestions);
                console.log('📋 Question count:', secondSurveyQuestions.length);
                console.log('📊 Config samples:', config.calibrationSamples);
                handleStartCalibration();
                setActiveTab('calibration');
              }}
              disabled={secondSurveyQuestions.length === 0}
              className="w-full bg-green-600 hover:bg-green-700 disabled:bg-slate-400 disabled:cursor-not-allowed text-white py-4 rounded-lg font-bold text-lg shadow-lg flex items-center justify-center gap-2"
            >
              <RefreshCw size={20} />
              Start Calibration Phase
            </button>
          </div>
        )}

        {/* Tab 5: Calibration */}
        {activeTab === 'calibration' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h2 className="text-2xl font-bold">Calibration Phase</h2>
                  <p className="text-slate-600">Each agent answers {config.calibrationSamples}× to determine response distributions</p>
                </div>
                {calibrationInProgress && (
                  <div className="text-sm text-slate-600 bg-slate-100 px-4 py-2 rounded-lg">
                    {calibrationProgress.toFixed(0)}% Complete
                  </div>
                )}
              </div>

              {calibrationInProgress && (
                <div className="mb-6">
                  <div className="w-full h-3 bg-slate-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-600 transition-all duration-300"
                      style={{ width: `${calibrationProgress}%` }}
                    />
                  </div>
                </div>
              )}

              <CalibrationTable />
            </div>

            {!calibrationInProgress && calibrationData.length > 0 && (
              <button
                onClick={() => setActiveTab('simulation')}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-4 rounded-lg font-bold text-lg shadow-lg"
              >
                Continue to Population Simulation →
              </button>
            )}
          </div>
        )}

        {/* Tab 6: Population Simulation */}
        {activeTab === 'simulation' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h2 className="text-2xl font-bold">Population Simulation</h2>
                  <p className="text-slate-600">Generate {config.simulatedPopulation.toLocaleString()} synthetic respondents using calibrated probabilities</p>
                </div>
              </div>

              {!simulationInProgress && !simulatedData && (
                <div className="text-center py-12 bg-slate-50 rounded-xl border-2 border-dashed">
                  <Users className="w-16 h-16 mx-auto text-slate-300 mb-4" />
                  <p className="text-slate-500 mb-4">Ready to simulate population</p>
                  <button
                    onClick={handleStartSimulation}
                    className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-bold text-lg shadow-lg flex items-center gap-2 mx-auto"
                  >
                    <Play size={20} />
                    Start Simulation
                  </button>
                </div>
              )}

              {simulationInProgress && (
                <div>
                  <div className="w-full h-3 bg-slate-200 rounded-full overflow-hidden mb-4">
                    <div 
                      className="h-full bg-green-600 transition-all duration-300"
                      style={{ width: `${simulationProgress}%` }}
                    />
                  </div>
                  <p className="text-center text-slate-600">Simulating... {simulationProgress.toFixed(0)}%</p>
                </div>
              )}

              {simulatedData && <SimulationSummary />}
            </div>

            {simulatedData && (
              <button
                onClick={() => setActiveTab('analysis')}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-4 rounded-lg font-bold text-lg shadow-lg"
              >
                Continue to Analysis & Downloads →
              </button>
            )}
          </div>
        )}

        {/* Tab 7: Analysis & Downloads */}
        {activeTab === 'analysis' && simulatedData && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-2xl font-bold mb-6">Analysis & Export</h2>
              
              <SimulationSummary />

              <div className="mt-6 space-y-3">
                <button 
                  onClick={() => handleDownload('/download/simulated-survey', 'simulated_survey.csv')}
                  className="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-bold flex items-center justify-center gap-2"
                >
                  <Download size={20} />
                  Download Simulated Population (CSV - {config.simulatedPopulation.toLocaleString()} respondents)
                </button>
                
                <button 
                  onClick={() => handleDownload('/download/calibration', 'calibration_data.csv')}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-bold flex items-center justify-center gap-2"
                >
                  <Database size={20} />
                  Download Calibration Data (CSV)
                </button>
                
                <button 
                  onClick={() => handleDownload('/download/personas', 'personas.json')}
                  className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-lg font-bold flex items-center justify-center gap-2"
                >
                  <FileText size={20} />
                  Download Personas (JSON)
                </button>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-bold">Simulated Population Visualization</h3>
                {!visualizationUrl && (
                  <button
                    onClick={handleGenerateVisualization}
                    disabled={isLoading}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium flex items-center gap-2"
                  >
                    <BarChart size={16} /> Generate Visualization
                  </button>
                )}
              </div>

              {visualizationUrl ? (
                <div className="border rounded-lg overflow-hidden">
                  <img src={visualizationUrl} alt="Simulated Population" className="w-full" />
                </div>
              ) : (
                <div className="border-2 border-dashed border-slate-300 rounded-lg p-12 text-center">
                  <BarChart className="w-16 h-16 mx-auto text-slate-300 mb-4" />
                  <p className="text-slate-500">Click "Generate Visualization" to create distribution plots</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}