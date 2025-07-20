// src/pages/Library.jsx
import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { Search, Play, Edit, Trash2, Filter } from 'lucide-react';
import { getPromptLibrary } from '../services/api';

const Library = () => {
  const [prompts, setPrompts] = useState([]);
  const [filteredPrompts, setFilteredPrompts] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [isLoading, setIsLoading] = useState(true);
  const pageRef = useRef(null);

  useEffect(() => {
    // Page entrance animation
    gsap.fromTo(pageRef.current,
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 1, ease: 'power2.out' }
    );

    loadLibrary();
  }, []);

  useEffect(() => {
    // Filter prompts based on search and category
    let filtered = prompts;

    if (searchTerm) {
      filtered = filtered.filter(prompt =>
        prompt.original_prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
        prompt.optimized_prompt.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (selectedCategory !== 'all') {
      filtered = filtered.filter(prompt => prompt.task_type === selectedCategory);
    }

    setFilteredPrompts(filtered);
  }, [prompts, searchTerm, selectedCategory]);

  const loadLibrary = async () => {
    try {
      const response = await getPromptLibrary();
      setPrompts(response.recent_prompts || []);
    } catch (error) {
      console.error('Failed to load library:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeploy = (prompt) => {
    // Placeholder for deploy functionality
    console.log('Deploy prompt:', prompt.id);
  };

  const handleOptimize = (prompt) => {
    // Navigate to optimize page with this prompt
    console.log('Optimize prompt:', prompt.id);
  };

  const categories = [
    { value: 'all', label: 'All Categories' },
    { value: 'content_creation', label: 'Content Creation' },
    { value: 'social_media', label: 'Social Media' },
    { value: 'email', label: 'Email' },
    { value: 'video_script', label: 'Video Script' },
    { value: 'prospecting', label: 'Prospecting' }
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-300">Loading your prompt library...</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={pageRef} className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-white">
          Prompt <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Library</span>
        </h1>
        <p className="text-gray-300 text-lg">Your collection of optimized prompts ready for action</p>
      </div>

      {/* Search and Filters */}
      <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Search */}
          <div className="md:col-span-2 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search prompts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </div>

          {/* Category Filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-400 appearance-none"
            >
              {categories.map(category => (
                <option key={category.value} value={category.value} className="bg-slate-800">
                  {category.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Prompts Grid */}
      {filteredPrompts.length === 0 ? (
        <div className="text-center py-16">
          <div className="w-24 h-24 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <Search className="text-gray-400" size={32} />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">No prompts found</h3>
          <p className="text-gray-400">Try adjusting your search terms or create some new prompts</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredPrompts.map((prompt, index) => (
            <PromptCard
              key={prompt.id}
              prompt={prompt}
              index={index}
              onDeploy={handleDeploy}
              onOptimize={handleOptimize}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const PromptCard = ({ prompt, index, onDeploy, onOptimize }) => {
  const cardRef = useRef(null);

  useEffect(() => {
    // Stagger animation for cards
    gsap.fromTo(cardRef.current,
      { y: 50, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.6, delay: index * 0.1, ease: 'power2.out' }
    );
  }, [index]);

  const handleCardHover = () => {
    gsap.to(cardRef.current, { scale: 1.02, duration: 0.3, ease: 'power2.out' });
  };

  const handleCardLeave = () => {
    gsap.to(cardRef.current, { scale: 1, duration: 0.3, ease: 'power2.out' });
  };

  return (
    <div
      ref={cardRef}
      className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 hover:border-white/40 transition-all duration-300"
      onMouseEnter={handleCardHover}
      onMouseLeave={handleCardLeave}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="px-3 py-1 bg-blue-500/20 text-blue-300 text-xs font-medium rounded-full">
          {prompt.task_type.replace('_', ' ').toUpperCase()}
        </span>
        <span className="text-gray-400 text-xs">
          {new Date(prompt.timestamp).toLocaleDateString()}
        </span>
      </div>

      {/* Content Preview */}
      <div className="space-y-3 mb-6">
        <div>
          <h4 className="text-gray-300 text-sm font-medium mb-1">Original:</h4>
          <p className="text-gray-200 text-sm line-clamp-2">{prompt.original_prompt}</p>
        </div>
        <div>
          <h4 className="text-gray-300 text-sm font-medium mb-1">Optimized:</h4>
          <p className="text-white text-sm line-clamp-2">{prompt.optimized_prompt}</p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex space-x-2">
        <button
          onClick={() => onDeploy(prompt)}
          className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white text-sm font-medium rounded-lg hover:from-green-600 hover:to-emerald-700 transition-all duration-300"
        >
          <Play size={16} />
          <span>Deploy</span>
        </button>
        <button
          onClick={() => onOptimize(prompt)}
          className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-white/10 border border-white/20 text-white text-sm font-medium rounded-lg hover:bg-white/20 transition-all duration-300"
        >
          <Edit size={16} />
          <span>Optimize</span>
        </button>
      </div>
    </div>
  );
};

export default Library;
