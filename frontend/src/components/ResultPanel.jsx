import { AlertTriangle, CheckCircle, Info } from "lucide-react";
import { cn } from "../lib/utils";

export function ResultPanel({ result }) {
  if (!result) return null;

  const { risk_level, flagged_for_review, decision_threshold, risk_score } = result;
  
  let theme = "bg-success-50 text-success-800 border-success-200";
  let Icon = CheckCircle;
  let iconColor = "text-success-600";
  
  if (flagged_for_review) {
    if (risk_level === "Very High") {
      theme = "bg-danger-50 text-danger-900 border-danger-200";
      iconColor = "text-danger-600";
    } else {
      theme = "bg-warning-50 text-warning-900 border-warning-200";
      iconColor = "text-warning-600";
    }
    Icon = AlertTriangle;
  }

  return (
    <div className={cn("rounded-2xl border p-5 animate-in mt-6", theme)}>
      <div className="flex items-start">
        <Icon className={cn("w-6 h-6 mr-3 mt-0.5 flex-shrink-0", iconColor)} />
        <div>
          <h4 className="text-lg font-bold mb-1">
            {flagged_for_review ? "Clinical Review Recommended" : "No Immediate Action Required"}
          </h4>
          <p className="text-sm opacity-90 leading-relaxed mb-4">
            {flagged_for_review 
              ? `The patient's risk score (${risk_score.toFixed(3)}) exceeds the clinical screening threshold of ${decision_threshold}. Follow-up HbA1c or FPG testing is advised.`
              : `The patient's risk profile falls below the clinical screening threshold. Continue standard preventative care.`}
          </p>
          
          <div className="bg-white/60 rounded-lg p-3 text-sm">
            <h5 className="font-semibold mb-2 flex items-center">
              <Info className="w-4 h-4 mr-1.5" />
              Suggested Actions:
            </h5>
            <ul className="list-disc pl-5 space-y-1 opacity-90">
              {flagged_for_review ? (
                <>
                  <li>Schedule fasting plasma glucose (FPG) test</li>
                  <li>Refer to lifestyle intervention program</li>
                  <li>Discuss cardiovascular risk factors</li>
                </>
              ) : (
                <>
                  <li>Maintain current healthy lifestyle habits</li>
                  <li>Re-assess in 12 months</li>
                </>
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
