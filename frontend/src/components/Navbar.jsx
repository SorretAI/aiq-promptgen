// src/components/Navbar.jsx
import React, { useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { gsap } from 'gsap';
import { Zap, BookOpen, BarChart3 } from 'lucide-react';

const Navbar = () => {
  const navRef = useRef(null);
  const logoRef = useRef(null);
  const location = useLocation();

  useEffect(() => {
    // Navbar entrance animation
    gsap.fromTo(navRef.current,
      { y: -100, opacity: 0 },
      { y: 0, opacity: 1, duration: 1, ease: 'power2.out', delay: 0.3 }
    );

    // Logo special animation
    gsap.fromTo(logoRef.current,
      { scale: 0.8, rotation: -10 },
      { scale: 1, rotation: 0, duration: 1.5, ease: 'elastic.out', delay: 0.5 }
    );
  }, []);

  const navItems = [
    { path: '/optimize', label: 'Optimize', icon: Zap },
    { path: '/library', label: 'Library', icon: BookOpen },
    { path: '/stats', label: 'Stats', icon: BarChart3 }
  ];

  return (
    <nav ref={navRef} className="relative z-20 backdrop-blur-sm bg-white/5 border-b border-white/10">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3">
            <div ref={logoRef} className="relative">
              <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-green-400 flex items-center justify-center text-white font-bold text-xl shadow-lg">
                Q
              </div>
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-green-400 animate-ping opacity-20"></div>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">AIQ</h1>
              <p className="text-xs text-gray-300 uppercase tracking-wider">Human Potential Amplification</p>
            </div>
          </Link>

          {/* Navigation */}
          <div className="flex space-x-8">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 hover:bg-white/10 hover:scale-105 ${
                  location.pathname === path
                    ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 text-blue-300 border border-blue-400/30'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                <Icon size={20} />
                <span className="font-medium">{label}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
