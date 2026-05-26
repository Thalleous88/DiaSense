import { Activity, LayoutDashboard, FileText, Settings, UserCircle, LogOut } from "lucide-react";

export function Sidebar() {
  return (
    <aside className="fixed inset-y-0 left-0 w-64 bg-white border-r border-slate-200 hidden lg:flex flex-col z-10">
      <div className="h-16 flex items-center px-6 border-b border-slate-200">
        <Activity className="w-6 h-6 text-primary-600 mr-2" />
        <span className="text-xl font-bold text-slate-900 tracking-tight">Dia<span className="text-primary-600">Sense</span></span>
      </div>
      
      <div className="p-4 flex-1">
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 mt-4 px-2">Menu</div>
        <nav className="space-y-1">
          <a href="#" className="flex items-center px-2 py-2.5 bg-primary-50 text-primary-700 rounded-lg group font-medium">
            <LayoutDashboard className="w-5 h-5 mr-3 text-primary-600" />
            Dashboard
          </a>
          <a href="#" className="flex items-center px-2 py-2.5 text-slate-600 hover:bg-slate-50 hover:text-slate-900 rounded-lg group font-medium transition-colors">
            <FileText className="w-5 h-5 mr-3 text-slate-400 group-hover:text-slate-600" />
            Assessments
          </a>
          <a href="#" className="flex items-center px-2 py-2.5 text-slate-600 hover:bg-slate-50 hover:text-slate-900 rounded-lg group font-medium transition-colors">
            <UserCircle className="w-5 h-5 mr-3 text-slate-400 group-hover:text-slate-600" />
            Patients
          </a>
        </nav>

        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 mt-8 px-2">Settings</div>
        <nav className="space-y-1">
          <a href="#" className="flex items-center px-2 py-2.5 text-slate-600 hover:bg-slate-50 hover:text-slate-900 rounded-lg group font-medium transition-colors">
            <Settings className="w-5 h-5 mr-3 text-slate-400 group-hover:text-slate-600" />
            Preferences
          </a>
        </nav>
      </div>

      <div className="p-4 border-t border-slate-200">
        <button className="flex items-center w-full px-2 py-2.5 text-slate-600 hover:bg-slate-50 hover:text-slate-900 rounded-lg group font-medium transition-colors">
          <LogOut className="w-5 h-5 mr-3 text-slate-400 group-hover:text-slate-600" />
          Log Out
        </button>
      </div>
    </aside>
  );
}
