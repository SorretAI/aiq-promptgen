// src/components/QInterface.jsx
import React, { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';

const QInterface = ({ prompt, setPrompt, onOptimize, isLoading }) => {
  const qRef = useRef(null);
  const containerRef = useRef(null);
  const [hasAnimated, setHasAnimated] = useState(false);

  useEffect(() => {
    if (!hasAnimated) {
      // Q Interface intro animation - only runs once
      const tl = gsap.timeline();
      
      tl.fromTo(qRef.current,
        { scale: 0, rotation: 360, opacity: 0 },
        { scale: 1, rotation: 0, opacity: 1, duration: 2, ease: 'elastic.out' }
      )
      .fromTo(containerRef.current.querySelector('.q-glow'),
        { scale: 0 },
        { scale: 1, duration: 1, ease: 'power2.out' },
        '-=1'
      )
      .fromTo(containerRef.current.querySelector('textarea'),
        { y: 50, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, ease: 'power2.out' },
        '-=0.5'
      );

      setHasAnimated(true);
    }
  }, [hasAnimated]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (prompt.trim() && !isLoading) {
      onOptimize();
    }
  };

  return (
    <div ref={containerRef} className="relative">
      {/* Q Symbol with Glow Effect */}
      <div className="flex justify-center mb-8">
        <div className="relative">
          <div 
            ref={qRef}
            className="w-24 h-24 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-green-400 flex items-center justify-center text-white font-bold text-4xl shadow-2xl"
          >
            Q
          </div>
          <div className="q-glow absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-green-400 blur-lg opacity-50 -z-10"></div>
        </div>
      </div>

      {/* Prompt Input */}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="relative">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt here to unlock its potential..."
            className="w-full h-32 p-6 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent transition-all duration-300"
            disabled={isLoading}
          />
          <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-green-400/10 -z-10 blur-sm"></div>
        </div>

        <button
          type="submit"
          disabled={!prompt.trim() || isLoading}
          className="w-full py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-xl transition-all duration-300 hover:from-blue-600 hover:to-purple-700 hover:scale-105 hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          {isLoading ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              <span>Optimizing...</span>
            </div>
          ) : (
            'Amplify Potential'
          )}
        </button>
      </form>
    </div>
  );
};

export default QInterface;
