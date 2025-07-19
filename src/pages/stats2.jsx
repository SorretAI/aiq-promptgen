// src/pages/Stats.jsx
import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { TrendingUp, Brain, Clock, Zap } from 'lucide-react';
import { getSystemStats } from '../services/api';

const Stats = () => {
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const pageRef = useRef(null);

  useEffect(() => {
    // Page entrance animation
    gsap.fromTo(pageRef.current,
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 1, ease: 'power2.out' }
    );

    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await getSystemStats();
      setStats(response);
    } catch (error) {
      console.error('Failed to load stats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-300">Loading system statistics...</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={pageRef} className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-white">
          System <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Statistics</span>
        </h1>
        <p className="text-gray-300 text-lg">Monitor your AI optimization performance and usage</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          icon={Brain}
          title="Available Models"
          value={stats?.available_models?.length || 0}
          subtitle="AI Models Ready"
          color="blue"
        />
        <StatsCard
          icon={TrendingUp}
          title="Prompt Library"
          value={stats?.prompt_library_size || 0}
          subtitle="Optimized Prompts"
          color="green"
        />
        <StatsCard
          icon={Zap}
          title="Active Sessions"
          value={stats?.active_sessions || 0}
          subtitle="Current Optimizations"
          color="purple"
        />
        <StatsCard
          icon={Clock}
          title="Learning Tasks"
          value={stats?.user_memory_tasks || 0}
          subtitle="Remembered Preferences"
          color="yellow"
        />
      </div>

      {/* System Status */}
      <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
        <h2 className="text-2xl font-bold text-white mb-6">System Status</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* AI Models */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-300">AI Models</h3>
            <div className="space-y-3">
              {stats?.available_models?.map((model, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                  <span className="text-white font-medium">{model}</span>
                  <span className="px-2 py-1 bg-green-500/20 text-green-300 text-xs rounded-full">
                    Active
                  </span>
                </div>
              )) || []}
            </div>
          </div>

          {/* System Health */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-300">System Health</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white">System Status</span>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  stats?.system_status === 'operational' 
                    ? 'bg-green-500/20 text-green-300' 
                    : 'bg-red-500/20 text-red-300'
                }`}>
                  {stats?.system_status || 'Unknown'}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white">Memory Usage</span>
                <span className="text-gray-300">Optimal</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white">Response Time</span>
                <span className="text-gray-300">< 2s</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatsCard = ({ icon: Icon, title, value, subtitle, color }) => {
  const cardRef = useRef(null);

  useEffect(() => {
    // Card entrance animation
    gsap.fromTo(cardRef.current,
      { y: 30, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out' }
    );

    // Value count-up animation
    gsap.fromTo(cardRef.current.querySelector('.stat-value'),
      { textContent: 0 },
      { 
        textContent: value,
        duration: 1.5,
        ease: 'power2.out',
        snap: { textContent: 1 },
        delay: 0.5
      }
    );
  }, [value]);

  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-400/30',
    green: 'from-green-500/20 to-green-600/20 border-green-400/30',
    purple: 'from-purple-500/20 to-purple-600/20 border-purple-400/30',
    yellow: 'from-yellow-500/20 to-yellow-600/20 border-yellow-400/30'
  };

  const iconColorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    purple: 'text-purple-400',
    yellow: 'text-yellow-400'
  };

  return (
    <div
      ref={cardRef}
      className={`bg-gradient-to-br ${colorClasses[color]} backdrop-blur-sm rounded-xl p-6 border`}
    >
      <div className="flex items-center justify-between mb-4">
        <Icon className={iconColorClasses[color]} size={32} />
      </div>
      <div className="space-y-2">
        <div className={`stat-value text-3xl font-bold text-white`}>{value}</div>
        <div className="text-gray-300 font-medium">{title}</div>
        <div className="text-gray-400 text-sm">{subtitle}</div>
      </div>
    </div>
  );
};

export default Stats;
