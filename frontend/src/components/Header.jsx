import { NavLink } from 'react-router-dom';
import { Activity, LayoutDashboard, Brain } from 'lucide-react';

export function Header() {
  const navLinkClass = ({ isActive }) =>
    `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? 'bg-primary-50 text-primary-700'
        : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
    }`;

  return (
    <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-20 flex items-center justify-between px-4 sm:px-6">
      <div className="flex items-center gap-6">
        <div className="flex items-center">
          <Activity className="w-6 h-6 text-primary-600 mr-2" />
          <span className="text-xl font-bold text-slate-900 tracking-tight">Dia<span className="text-primary-600">Sense</span></span>
        </div>
        <nav className="hidden sm:flex items-center gap-1">
          <NavLink to="/" end className={navLinkClass}>
            {({ isActive }) => (
              <>
                <LayoutDashboard className="w-4 h-4" />
                Dashboard
              </>
            )}
          </NavLink>
          <NavLink to="/training" className={navLinkClass}>
            {({ isActive }) => (
              <>
                <Brain className="w-4 h-4" />
                Model Training
              </>
            )}
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
