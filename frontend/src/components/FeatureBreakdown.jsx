export function FeatureBreakdown() {
  const features = [
    { name: "General Health", impact: 85, type: "negative" },
    { name: "High Blood Pressure", impact: 72, type: "negative" },
    { name: "BMI", impact: 68, type: "negative" },
    { name: "Age", impact: 55, type: "neutral" },
    { name: "Physical Activity", impact: 40, type: "positive" },
  ];

  return (
    <div className="glass-card p-5 mt-6">
      <h3 className="text-lg font-semibold text-slate-900 mb-4">Risk Factor Breakdown</h3>
      <div className="space-y-4">
        {features.map((feature, i) => (
          <div key={i}>
            <div className="flex justify-between text-sm mb-1.5">
              <span className="font-medium text-slate-700">{feature.name}</span>
              <span className="text-slate-500">{feature.impact}% impact</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div 
                className={`h-2 rounded-full ${
                  feature.type === 'negative' ? 'bg-danger-500' : 
                  feature.type === 'positive' ? 'bg-success-500' : 'bg-warning-500'
                }`}
                style={{ width: `${feature.impact}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
      <p className="text-xs text-slate-400 mt-4 italic">
        * Feature importance based on SHAP values from the XGBoost model.
      </p>
    </div>
  );
}
