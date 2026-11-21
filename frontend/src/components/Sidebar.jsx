import React from 'react';
import {
  LayoutDashboard,
  Database,
  Zap,
  Activity,
  BarChart3,
  Sliders,
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, isOpen, onClose }) {
  const navSections = [
    {
      title: 'MAIN',
      items: [
        { id: 'overview', label: 'Overview', icon: LayoutDashboard },
        { id: 'datatrain', label: 'Dataset & Training', icon: Database },
        { id: 'livedemo', label: 'Live Demo', icon: Zap },
        { id: 'benchmarking', label: 'Benchmarking', icon: Activity },
        { id: 'results', label: 'Results & Comparison', icon: BarChart3 },
        { id: 'settings', label: 'Settings', icon: Sliders },
      ],
    },
  ];

  const handleClick = (id) => {
    setActiveTab(id);
    if (onClose) onClose();
  };

  return (
    <aside
      className={`fixed inset-y-0 left-0 w-64 bg-white border-r border-slate-200 transform ${isOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-200 ease-in-out md:relative md:translate-x-0 md:z-auto z-20`}
    >
      {/* Brand Header */}
      <div className="p-4 border-b border-gray-300 flex items-center space-x-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg">
          <LayoutDashboard className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="font-bold text-sm tracking-wide text-primary flex items-center gap-1.5">
            ADAPT-EDGE <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-300 font-mono">v1.0</span>
          </div>
          <p className="text-[11px] text-primary truncate max-w-[150px]">Edge AI Switching Platform</p>
        </div>
      </div>

      {/* Nav Menu */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {navSections.map((sec, idx) => (
          <div key={idx} className="space-y-1">
            <div className="px-2 text-[10px] font-mono uppercase tracking-wider text-primary font-semibold">
              {sec.title}
            </div>
            {sec.items.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleClick(item.id)}
                  className={`w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-gradient-to-r from-cyan-500/15 to-indigo-500/10 text-cyan-300 border border-cyan-500/30 shadow-sm'
                      : 'text-primary hover:text-gray-800 hover:bg-gray-100 border border-transparent'
                  }`}
                >
                  <div className="flex items-center space-x-2.5 truncate">
                    <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-cyan-400' : 'text-primary'}`} />
                    <span className="truncate">{item.label}</span>
                  </div>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </aside>
  );
}
