import { Clock, ChevronRight } from "lucide-react";

export function RecentAssessments() {
  const assessments = [
    { id: "PAT-8829", date: "Today, 09:41 AM", score: 82.4, flag: true },
    { id: "PAT-7102", date: "Yesterday, 14:22", score: 12.1, flag: false },
    { id: "PAT-6641", date: "Oct 24, 11:05", score: 45.8, flag: true },
  ];

  return (
    <div className="glass-card p-5 mt-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-900 flex items-center">
          <Clock className="w-4 h-4 mr-2 text-slate-500" />
          Recent Assessments
        </h3>
        <button className="text-sm font-medium text-primary-600 hover:text-primary-700">View All</button>
      </div>
      
      <div className="space-y-3">
        {assessments.map((a, i) => (
          <div key={i} className="flex items-center justify-between p-3 rounded-xl border border-slate-100 hover:border-primary-100 hover:bg-primary-50/50 transition-colors cursor-pointer group">
            <div>
              <p className="font-medium text-slate-900 group-hover:text-primary-700 transition-colors">{a.id}</p>
              <p className="text-xs text-slate-500">{a.date}</p>
            </div>
            <div className="flex items-center">
              <div className="text-right mr-3">
                <p className={`font-bold ${a.flag ? 'text-danger-600' : 'text-success-600'}`}>
                  {a.score}%
                </p>
              </div>
              <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-primary-400" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
