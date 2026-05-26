import { cn } from "../lib/utils";

export function MetricCard({ title, value, subtitle, icon: Icon, trend = "neutral" }) {
  let trendColor = "text-slate-500";
  if (trend === "up") trendColor = "text-danger-500";
  if (trend === "down") trendColor = "text-success-500";

  return (
    <div className="glass-card p-5 premium-shadow transition-transform hover:-translate-y-1 duration-300">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-slate-500 mb-1">{title}</p>
          <h4 className="text-2xl font-bold text-slate-900 tracking-tight">{value}</h4>
        </div>
        <div className={cn("p-2 rounded-lg bg-slate-50", trendColor)}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
      {subtitle && (
        <p className="text-xs text-slate-400 mt-3 flex items-center">
          {subtitle}
        </p>
      )}
    </div>
  );
}
