// src/pages/Optimize.jsx
import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import QInterface from '../components/QInterface';
import OptimizationWizard from '../components/OptimizationWizard';
import ResultsDisplay from '../components/ResultsDisplay';
import { optimizePrompt } from '../services/api';

const Optimize = () => {
  const [prompt, setPrompt] = useState('');
  const [optimizationLevel, setOptimizationLevel] = useState('standard');
  const [taskType, setTaskType] = useState('content_creation');
  const [showWizard, setShowWizard] = useState(false);
  const [wizardSession, setWizardSession] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const pageRef = useRef(null);

  useEffect(() => {
    // Page entrance animation
    gsap.fromTo(pageRef.current,
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 1, ease: 'power2.out' }
    );
  }, []);

  const handleOptimize = async () => {
    setIsLoading(true);
    setResult(null);

    try {
      const response = await optimizePrompt({
        original_prompt: prompt,
        task_type: taskType,
        optimization_level: optimizationLevel
      });

      if (optimizationLevel === 'low_find_tweaks' && response.session_id) {
        setWizardSession(response);
        setShowWizard(true);
      } else {
        setResult(response);
      }
    } catch (error) {
      console.error('Optimization failed:', error);
      // Handle error - show toast or error message
    } finally {
      setIsLoading(false);
    }
  };

  const handleWizardComplete = (result) => {
    setShowWizard(false);
    setWizardSession(null);
    setResult(result);
  };

  return (
    <div ref={pageRef} className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-white">
          Optimize Your <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Prompts</span>
        </h1>
        <p className="text-gray-300 text-lg">Transform your ideas into powerful, actionable prompts</p>
      </div>

      {!showWizard && !result && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Q Interface */}
          <div className="lg:col-span-2">
            <QInterface
              prompt={prompt}
              setPrompt={setPrompt}
              onOptimize={handleOptimize}
              isLoading={isLoading}
            />
          </div>

          {/* Optimization Controls */}
          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <h3 className="text-white font-semibold mb-4">Optimization Level</h3>
              <div className="space-y-3">
                {[
                  { value: 'low_regenerate', label: 'Regenerate', desc: 'Create a variant' },
                  { value: 'low_find_tweaks', label: 'Find Tweaks', desc: 'Interactive optimization' },
                  { value: 'standard', label: 'Standard', desc: 'Direct optimization' },
                  { value: 'advanced', label: 'Advanced', desc: 'Deep analysis' }
                ].map((level) => (
                  <label key={level.value} className="flex items-center space-x-3 cursor-pointer group">
                    <input
                      type="radio"
                      name="optimization_level"
                      value={level.value}
                      checked={optimizationLevel === level.value}
                      onChange={(e) => setOptimizationLevel(e.target.value)}
                      className="w-4 h-4 text-blue-500 focus:ring-blue-400"
                    />
                    <div className="flex-1">
                      <div className="text-white group-hover:text-blue-300 transition-colors">{level.label}</div>
                      <div className="text-xs text-gray-400">{level.desc}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <h3 className="text-white font-semibold mb-4">Task Type</h3>
              <select
                value={taskType}
                onChange={(e) => setTaskType(e.target.value)}
                className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                <option value="content_creation">Content Creation</option>
                <option value="social_media">Social Media</option>
                <option value="email">Email</option>
                <option value="video_script">Video Script</option>
                <option value="prospecting">Prospecting</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Wizard Modal */}
      {showWizard && wizardSession && (
        <OptimizationWizard
          session={wizardSession}
          originalPrompt={prompt}
          onComplete={handleWizardComplete}
          onCancel={() => setShowWizard(false)}
        />
      )}

      {/* Results Display */}
      {result && (
        <ResultsDisplay
          result={result}
          onSaveToLibrary={() => {}}
          onUpdateMemory={() => {}}
          onOptimizeAgain={() => setResult(null)}
        />
      )}
    </div>
  );
};

export default Optimize;
