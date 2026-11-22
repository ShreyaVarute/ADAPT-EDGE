import React, { useState, Component } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';

// Error Boundary to prevent full black/blank screens if a component crashes
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 max-w-4xl mx-auto space-y-4">
          <div className="p-6 rounded-xl border border-red-200 bg-red-50 text-red-900 space-y-3">
            <h2 className="text-lg font-bold flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-red-600 inline-block animate-pulse"></span>
              View Render Error
            </h2>
            <p className="text-xs text-red-700 font-mono">
              {this.state.error && this.state.error.toString()}
            </p>
            {this.state.errorInfo && (
              <pre className="p-3 bg-white/80 rounded-lg border border-red-100 text-[11px] overflow-auto max-h-60 text-slate-700">
                {this.state.errorInfo.componentStack}
              </pre>
            )}
            <button
              onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-semibold cursor-pointer"
            >
              Retry Render
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// Pages
import Overview from './pages/Overview';
import Novelties from './pages/Novelties';
import DataTrain from './pages/DataTrain';
import LiveDemo from './pages/LiveDemo';
import Benchmarking from './pages/Benchmarking';
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
      case 'novelties':
        return <Novelties key={refreshKey} />;
      case 'datatrain':
        return <DataTrain key={refreshKey} />;
      case 'livedemo':
        return <LiveDemo key={refreshKey} />;
      case 'benchmarking':
        return <Benchmarking key={refreshKey} />;
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
          <ErrorBoundary key={activeTab}>
            {renderActivePage()}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}

