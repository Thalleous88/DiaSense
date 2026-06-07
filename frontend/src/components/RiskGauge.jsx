import { cn } from "../lib/utils";

export function RiskGauge({ score = 0, level = "Low", percentage = 0 }) {
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  let colorClass = "text-success-500";
  let bgGradient = "from-success-500/10 to-transparent";
  
  if (level === "Moderate") {
    colorClass = "text-warning-500";
    bgGradient = "from-warning-500/10 to-transparent";
  } else if (level === "High" || level === "Very High") {
    colorClass = "text-danger-500";
    bgGradient = "from-danger-500/10 to-transparent";
  }

  return (
    <div className={cn("glass-card p-6 flex flex-col items-center justify-center relative overflow-hidden bg-gradient-to-b", bgGradient)}>
      <h3 className="text-slate-500 font-medium text-sm mb-4">Current Risk Score</h3>
      
      <div className="relative flex items-center justify-center w-40 h-40">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 140 140">
          <circle
            cx="70"
            cy="70"
            r={radius}
            stroke="currentColor"
            strokeWidth="12"
            fill="transparent"
            className="text-slate-100"
          />
          <circle
            cx="70"
            cy="70"
            r={radius}
            stroke="currentColor"
            strokeWidth="12"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className={cn("transition-all duration-1000 ease-out", colorClass)}
          />
        </svg>
        
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-slate-900 tracking-tighter">
            {percentage}%
          </span>
          <span className={cn("text-xs font-semibold mt-1 px-2 py-0.5 rounded-full bg-white shadow-sm border border-slate-100", colorClass)}>
            {level}
          </span>
        </div>
      </div>
      
      <p className="text-center mt-4 text-sm text-slate-600 max-w-[200px]">
        Estimated probability of diabetes or prediabetes based on the submitted health indicators.
      </p>
    </div>
  );
}
