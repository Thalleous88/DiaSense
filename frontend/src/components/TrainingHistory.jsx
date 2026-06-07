import { Clock, CheckCircle, XCircle, Loader2 } from 'lucide-react';

export function TrainingHistory({ history }) {
  if (!history || history.length === 0) {
    return (
      <div className="glass-card p-5">
        <h3 className="text-lg font-semibold text-slate-900 mb-2 flex items-center">
          <Clock className="w-4 h-4 mr-2 text-slate-500" />
          Training History
        </h3>
        <p className="text-sm text-slate-400">No training runs recorded yet.</p>
      </div>
    );
  }

  const statusIcon = (status) => {
    if (status === 'completed') return <CheckCircle className="w-4 h-4 text-success-600" />;
    if (status === 'failed') return <XCircle className="w-4 h-4 text-danger-600" />;
    if (status === 'running') return <Loader2 className="w-4 h-4 text-primary-600 animate-spin" />;
    return <Clock className="w-4 h-4 text-slate-400" />;
  };

  const statusBadge = (status) => {
    const colors = {
      completed: 'bg-success-100 text-success-800',
      failed: 'bg-danger-100 text-danger-800',
      running: 'bg-primary-100 text-primary-800',
      pending: 'bg-slate-100 text-slate-600',
    };
    return colors[status] || colors.pending;
  };

  return (
    <div className="glass-card p-5">
      <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
        <Clock className="w-4 h-4 mr-2 text-slate-500" />
        Training History
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="text-left py-2 px-2 text-slate-500 font-medium">Date</th>
              <th className="text-left py-2 px-2 text-slate-500 font-medium">New Samples</th>
              <th className="text-left py-2 px-2 text-slate-500 font-medium">Total</th>
              <th className="text-left py-2 px-2 text-slate-500 font-medium">ROC-AUC</th>
              <th className="text-left py-2 px-2 text-slate-500 font-medium">Recall</th>
              <th className="text-left py-2 px-2 text-slate-500 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {history.map((run) => (
              <tr key={run.id} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="py-2.5 px-2 text-slate-700">
                  {run.created_at ? new Date(run.created_at).toLocaleDateString() : '—'}
                </td>
                <td className="py-2.5 px-2 text-slate-700">{run.n_new_samples?.toLocaleString() || '—'}</td>
                <td className="py-2.5 px-2 text-slate-700">{run.n_total_samples?.toLocaleString() || '—'}</td>
                <td className="py-2.5 px-2 text-slate-700">{run.roc_auc?.toFixed(4) || '—'}</td>
                <td className="py-2.5 px-2 text-slate-700">{run.recall_class1?.toFixed(4) || '—'}</td>
                <td className="py-2.5 px-2">
                  <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${statusBadge(run.status)}`}>
                    {statusIcon(run.status)}
                    {run.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
