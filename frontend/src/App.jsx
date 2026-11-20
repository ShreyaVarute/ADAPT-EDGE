import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';

// Pages
import Overview from './pages/Overview';
import DataTrain from './pages/DataTrain';
import LiveDemo from './pages/LiveDemo';
import Results from './pages/Results';
import Settings from './pages/Settings';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshKey, setRefreshKey] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleRefresh = () => {
    setRefreshKey((k) => k + 1);
  };

  const renderActivePage = () => {
    switch (activeTab) {
      case 'overview':
        return <Overview key={refreshKey} setActiveTab={setActiveTab} />;
      case 'datatrain':
        return <DataTrain key={refreshKey} />;
      case 'livedemo':
        return <LiveDemo key={refreshKey} />;
      case 'results':
        return <Results key={refreshKey} />;
      case 'settings':
        return <Settings key={refreshKey} />;
      default:
        return <Overview key={refreshKey} setActiveTab={setActiveTab} />;
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 text-slate-900 font-sans">
      {/* Persistent Sidebar */}
      {sidebarOpen && (
        <button
          aria-label="Close navigation"
          className="fixed inset-0 z-10 bg-slate-900/20 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Navbar */}
        <Navbar activeTab={activeTab} onRefresh={handleRefresh} onToggleSidebar={() => setSidebarOpen((value) => !value)} />

        {/* Dynamic Page View with Custom Scroll */}
        <main className="flex-1 overflow-y-auto bg-slate-50">
          {renderActivePage()}
        </main>
      </div>
    </div>
  );
}

