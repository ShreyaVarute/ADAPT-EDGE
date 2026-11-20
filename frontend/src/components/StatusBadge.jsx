import React from 'react';

export default function StatusBadge({ status, label, size = 'sm' }) {
  const norm = (status || '').toUpperCase();
  let color = 'bg-gray-800 text-gray-300 border-gray-700';

  if (norm === 'PASS' || norm === 'HEALTHY' || norm === 'OPTIMAL' || norm === 'RETAIN' || norm.includes('CONFIDENT')) {
    color = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  } else if (norm === 'WARN' || norm === 'CAUTION' || norm === 'PROBE' || norm.includes('PROBE')) {
    color = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
  } else if (norm === 'FAIL' || norm === 'ERROR' || norm === 'EXHAUSTED' || norm === 'VIOLATION') {
    color = 'bg-rose-500/10 text-rose-400 border-rose-500/30';
  } else if (norm === 'SWITCH' || norm === 'MODEL_B' || norm === 'ACTIVE') {
    color = 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30';
  } else if (norm === 'HIT') {
    color = 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30';
  }

  const sizeClasses = size === 'xs' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2.5 py-1 text-xs';

  return (
    <span className={`inline-flex items-center font-mono font-medium rounded-full border ${color} ${sizeClasses}`}>
      {label || status}
    </span>
  );
}

