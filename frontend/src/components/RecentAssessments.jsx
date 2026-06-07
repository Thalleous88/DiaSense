import { useState, useEffect } from 'react';
import { Clock, ChevronRight } from "lucide-react";
import { apiFetch } from '../hooks/useApi';

export function RecentAssessments() {
  const [assessments, setAssessments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiFetch('/assessments/recent?limit=5')
      .then(data => setAssessments(data))
      .catch(() => setAssessments([]))
      .finally(() => setLoading(false));
  }, []);

  const formatDate = (dateStr) => {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    const now = new Date();
    const diffMs = now - d;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString();
  };

  return (
    <div className="glass-card p-5 mt-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-900 flex items-center">
          <Clock className="w-4 h-4 mr-2 text-slate-500" />
          Recent Assessments
        </h3>
      </div>
      
      {loading ? (
        <div className="space-y-3 animate-pulse">
          {[1,2,3].map(i => (
            <div key={i} className="h-14 bg-slate-100 rounded-xl" />
          ))}
        </div>
      ) : assessments.length === 0 ? (
        <p className="text-sm text-slate-400 text-center py-4">No assessments yet. Submit a prediction to see results here.</p>
      ) : (
        <div className="space-y-3">
          {assessments.map((a, i) => (
            <div key={i} className="flex items-center justify-between p-3 rounded-xl border border-slate-100 hover:border-primary-100 hover:bg-primary-50/50 transition-colors cursor-pointer group">
              <div>
                <p className="font-medium text-slate-900 group-hover:text-primary-700 transition-colors">{a.patient_id}</p>
                <p className="text-xs text-slate-500">{formatDate(a.created_at)}</p>
              </div>
              <div className="flex items-center">
                <div className="text-right mr-3">
                  <p className={`font-bold ${a.flagged_for_review ? 'text-danger-600' : 'text-success-600'}`}>
                    {a.risk_percentage}%
                  </p>
                  <p className="text-xs text-slate-400">{a.risk_level}</p>
                </div>
                <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-primary-400" />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
