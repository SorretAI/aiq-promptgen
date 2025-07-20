// src/components/OptimizationWizard.jsx
import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ArrowLeft, ArrowRight, Shuffle } from 'lucide-react';
import { optimizePrompt } from '../services/api';

const OptimizationWizard = ({ session, originalPrompt, onComplete, onCancel }) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const wizardRef = useRef(null);
  const questionRef = useRef(null);

  useEffect(() => {
    // Wizard entrance animation
    gsap.fromTo(wizardRef.current,
      { scale: 0.8, opacity: 0 },
      { scale: 1, opacity: 1, duration: 0.5, ease: 'power2.out' }
    );
  }, []);

  useEffect(() => {
    // Question transition animation
    if (questionRef.current) {
      gsap.fromTo(questionRef.current,
        { x: 30, opacity: 0 },
        { x: 0, opacity: 1, duration: 0.3, ease: 'power2.out' }
      );
    }
  }, [currentQuestionIndex]);

  const currentQuestion = session.questions[currentQuestionIndex];
  const isLastQuestion = currentQuestionIndex === session.questions.length - 1;

  const handleAnswerChange = (questionId, value) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value
    }));
  };

  const handleNext = () => {
    if (!isLastQuestion) {
      setCurrentQuestionIndex(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1);
    }
  };

  const handleRandomize = () => {
    // Placeholder for randomize functionality
    console.log('Randomize selected - will implement later');
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);

    try {
      // Format answers as "1.a 2.b,c 3.custom answer"
      const formattedAnswers = Object.entries(answers)
        .map(([questionId, answer]) => `${questionId}.${answer}`)
        .join(' ');

      const result = await optimizePrompt({
        original_prompt: originalPrompt,
        task_type: 'social_media', // This should come from parent
        optimization_level: 'low_find_tweaks',
        session_id: session.session_id,
        user_answers: formattedAnswers
      });

      onComplete(result);
    } catch (error) {
      console.error('Failed to submit answers:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
  <div ref={wizardRef} className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl border border-white/20 w-full max-w-4xl h-[85vh] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white">Optimization Wizard</h2>
              <p className="text-gray-300 text-sm mt-1">Question {currentQuestionIndex + 1} of {session.questions.length}</p>
            </div>
            <button
              onClick={onCancel}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Progress Bar */}
          <div className="mt-4 w-full bg-white/10 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentQuestionIndex + 1) / session.questions.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Original Prompt Reference */}
        <div className="p-4 bg-white/5 border-b border-white/10">
          <p className="text-gray-300 text-sm">Original prompt:</p>
          <p className="text-white text-sm mt-1 italic">"{originalPrompt.slice(0, 100)}..."</p>
        </div>

        {/* Question Content */}
        <div ref={questionRef} className="p-6 space-y-6 overflow-y-auto flex-1">
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">{currentQuestion.question}</h3>
            
            {/* Randomize Option (always first) */}
            <div className="mb-6">
              <button
                onClick={handleRandomize}
                className="flex items-center space-x-2 p-3 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-400/30 rounded-lg text-yellow-300 hover:bg-yellow-500/30 transition-all duration-300 w-full"
              >
                <Shuffle size={20} />
                <span>Randomize (Let AI decide)</span>
              </button>
            </div>

            {/* Answer Options */}
            <div className="space-y-3">
              {currentQuestion.options.map((option, index) => {
                const optionLetter = String.fromCharCode(97 + index); // a, b, c, d...
                const isSelected = answers[currentQuestion.id] === optionLetter;
                
                return (
                  <label
                    key={index}
                    className={`flex items-center space-x-3 p-4 rounded-lg border cursor-pointer transition-all duration-300 ${
                      isSelected
                        ? 'bg-blue-500/20 border-blue-400 text-blue-300'
                        : 'bg-white/5 border-white/20 text-gray-300 hover:bg-white/10 hover:border-white/30'
                    }`}
                  >
                    <input
                      type="radio"
                      name={`question_${currentQuestion.id}`}
                      value={optionLetter}
                      checked={isSelected}
                      onChange={(e) => handleAnswerChange(currentQuestion.id, e.target.value)}
                      className="w-4 h-4 text-blue-500 focus:ring-blue-400"
                    />
                    <span>{option}</span>
                  </label>
                );
              })}
            </div>

            {/* Custom Answer */}
            {currentQuestion.allow_custom && (
              <div className="mt-4">
                <label className="block text-gray-300 text-sm mb-2">Or provide a custom answer:</label>
                <textarea
                  placeholder="Type your custom answer here..."
                  value={answers[currentQuestion.id]?.startsWith('custom:') ? answers[currentQuestion.id].slice(7) : ''}
                  onChange={(e) => handleAnswerChange(currentQuestion.id, `custom:${e.target.value}`)}
                  className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
                  rows={3}
                />
              </div>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-6 border-t border-white/10 flex justify-between">
          <button
            onClick={handlePrevious}
            disabled={currentQuestionIndex === 0}
            className="flex items-center space-x-2 px-4 py-2 text-gray-400 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ArrowLeft size={20} />
            <span>Previous</span>
          </button>

          {isLastQuestion ? (
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || !answers[currentQuestion.id]}
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-lg hover:from-green-600 hover:to-emerald-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Optimizing...</span>
                </>
              ) : (
                <span>Complete Optimization</span>
              )}
            </button>
          ) : (
            <button
              onClick={handleNext}
              disabled={!answers[currentQuestion.id]}
              className="flex items-center space-x-2 px-4 py-2 text-blue-400 hover:text-blue-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span>Next</span>
              <ArrowRight size={20} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default OptimizationWizard;