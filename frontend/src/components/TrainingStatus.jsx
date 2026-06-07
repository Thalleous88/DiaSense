import { CheckCircle, Loader2, XCircle, Clock } from 'lucide-react';
import { cn } from '../lib/utils';

export function TrainingStatus({ status }) {
  const { status: state, progress, message } = status;

  let Icon = Clock;
  let iconColor = 'text-slate-400';
  let bgColor = 'bg-slate-50';
  let borderColor = 'border-slate-200';
  let statusText = 'Idle';
  let statusColor = 'text-slate-600';

  if (state === 'running') {
    Icon = Loader2;
    iconColor = 'text-primary-600';
    bgColor = 'bg-primary-50';
    borderColor = 'border-primary-200';
    statusText = 'Training in Progress';
    statusColor = 'text-primary-700';
  } else if (state === 'completed') {
    Icon = CheckCircle;
    iconColor = 'text-success-600';
    bgColor = 'bg-success-500/5';
    borderColor = 'border-success-500/30';
    statusText = 'Training Complete';
    statusColor = 'text-success-700';
  } else if (state === 'failed') {
    Icon = XCircle;
    iconColor = 'text-danger-600';
    bgColor = 'bg-danger-500/5';
    borderColor = 'border-danger-500/30';
    statusText = 'Training Failed';
    statusColor = 'text-danger-700';
  }

  return (
    <div className={cn('glass-card p-5 border', borderColor, bgColor)}>
      <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
        <Clock className="w-4 h-4 mr-2 text-slate-500" />
        Training Status
      </h3>
      <div className="flex items-center mb-3">
        <Icon className={cn('w-5 h-5 mr-2', iconColor, state === 'running' && 'animate-spin')} />
        <span className={cn('font-semibold text-sm', statusColor)}>{statusText}</span>
      </div>
      {state === 'running' && (
        <div className="mb-3">
          <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-primary-600 h-2.5 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-1">{progress}% — {message}</p>
        </div>
      )}
      {message && state !== 'running' && (
        <p className="text-sm text-slate-600">{message}</p>
      )}
      {state === 'idle' && (
        <p className="text-sm text-slate-500">No training in progress. Upload data to begin.</p>
      )}
    </div>
  );
}
