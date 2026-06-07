import { Brain } from 'lucide-react';

export function ModelInfo({ info }) {
  if (!info) {
    return (
      <div className="glass-card p-5">
        <h3 className="text-lg font-semibold text-slate-900 mb-2 flex items-center">
          <Brain className="w-4 h-4 mr-2 text-slate-500" />
          Current Model Info
        </h3>
        <p className="text-sm text-slate-400">Loading model information...</p>
      </div>
    );
  }

  const metrics = [
    { label: 'Model Type', value: info.model_type || '—' },
    { label: 'Base Models', value: (info.base_models || []).join(', ') },
    { label: 'Features', value: info.n_features || '—' },
    { label: 'Decision Threshold', value: info.decision_threshold?.toFixed(4) || '—' },
    { label: 'ROC-AUC', value: info.roc_auc?.toFixed(4) || '—' },
    { label: 'Recall (Class 1)', value: info.recall_class1?.toFixed(4) || '—' },
    { label: 'PR-AUC', value: info.pr_auc?.toFixed(4) || '—' },
    { label: 'F1 Score', value: info.f1_score?.toFixed(4) || '—' },
    { label: 'Training Date', value: info.training_date ? new Date(info.training_date).toLocaleDateString() : '—' },
    { label: 'Training Samples', value: info.n_training_samples?.toLocaleString() || '—' },
  ];

  return (
    <div className="glass-card p-5">
      <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
        <Brain className="w-4 h-4 mr-2 text-slate-500" />
        Current Model Info
      </h3>
      <div className="space-y-2.5">
        {metrics.map((m) => (
          <div key={m.label} className="flex justify-between items-center py-1 border-b border-slate-100 last:border-0">
            <span className="text-sm text-slate-500">{m.label}</span>
            <span className="text-sm font-medium text-slate-900 text-right max-w-[60%] truncate">{m.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
