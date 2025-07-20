// src/components/ResultsDisplay.jsx
import React, { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { Save, RefreshCw, Brain, Clock, Star } from 'lucide-react';

const ResultsDisplay = ({ result, onSaveToLibrary, onUpdateMemory, onOptimizeAgain }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    // Results entrance animation
    gsap.fromTo(containerRef.current,
      { y: 50, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.8, ease: 'power2.out' }
    );

    // Stagger animation for cards
    gsap.fromTo('.result-card',
      { y: 30, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.6, stagger: 0.2, ease: 'power2.out', delay: 0.3 }
    );
  }, []);

  return (
    <div ref={containerRef} className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-2">Optimization Complete</h2>
        <p className="text-gray-300">Your prompt has been enhanced and amplified</p>
      </div>

      {/* Before/After Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Original */}
        <div className="result-card bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
          <div className="flex items-center space-x-2 mb-4">
            <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
            <h3 className="text-gray-300 font-semibold">Original Prompt</h3>
          </div>
          <p className="text-gray-200 leading-relaxed">{result.original_prompt}</p>
        </div>

        {/* Optimized */}
        <div className="result-card bg-gradient-to-br from-blue-500/20 to-purple-500/20 backdrop-blur-sm rounded-xl p-6 border border-blue-400/30">
          <div className="flex items-center space-x-2 mb-4">
            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            <h3 className="text-blue-300 font-semibold">Optimized Prompt</h3>
          </div>
          <p className="text-white leading-relaxed">{result.optimized_prompt}</p>
        </div>
      </div>

      {/* Stats */}
      <div className="result-card grid grid-cols-1 md:grid-cols-4 gap-6 bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
        <div className="text-center">
          <div className="flex items-center justify-center w-12 h-12 bg-blue-500/20 rounded-full mx-auto mb-2">
            <Star className="text-blue-400" size={24} />
          </div>
          <div className="text-2xl font-bold text-white">{result.improvement_score.toFixed(1)}</div>
          <div className="text-sm text-gray-400">Improvement Score</div>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center w-12 h-12 bg-purple-500/20 rounded-full mx-auto mb-2">
            <Brain className="text-purple-400" size={24} />
          </div>
          <div className="text-2xl font-bold text-white">{result.model_used}</div>
          <div className="text-sm text-gray-400">AI Model</div>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center w-12 h-12 bg-green-500/20 rounded-full mx-auto mb-2">
            <Clock className="text-green-400" size={24} />
          </div>
          <div className="text-2xl font-bold text-white">{new Date(result.timestamp).toLocaleDateString()}</div>
          <div className="text-sm text-gray-400">Created</div>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center w-12 h-12 bg-yellow-500/20 rounded-full mx-auto mb-2">
            <Save className="text-yellow-400" size={24} />
          </div>
          <div className="text-2xl font-bold text-white">{result.stored_in_library ? 'Yes' : 'No'}</div>
          <div className="text-sm text-gray-400">In Library</div>
        </div>
      </div>

      {/* Actions */}
      <div className="result-card flex flex-wrap gap-4 justify-center">
        <button
          onClick={onSaveToLibrary}
          className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-lg hover:from-green-600 hover:to-emerald-700 transition-all duration-300 hover:scale-105"
        >
          <Save size={20} />
          <span>Save to Library</span>
        </button>

        {result.memory_updated && (
          <button
            onClick={onUpdateMemory}
            className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold rounded-lg hover:from-purple-600 hover:to-indigo-700 transition-all duration-300 hover:scale-105"
          >
            <Brain size={20} />
            <span>Update Memory</span>
          </button>
        )}

        <button
          onClick={onOptimizeAgain}
          className="flex items-center space-x-2 px-6 py-3 bg-white/10 backdrop-blur-sm border border-white/20 text-white font-semibold rounded-lg hover:bg-white/20 transition-all duration-300 hover:scale-105"
        >
          <RefreshCw size={20} />
          <span>Optimize Again</span>
        </button>
      </div>
    </div>
  );
};

export default ResultsDisplay;
