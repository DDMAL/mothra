import { useState } from "react";
import type { CurrentUser } from "../../hooks/useAuth";

interface NavbarProps {
  onLogin?: () => void;
  onGetStarted?: () => void;
  onMyProjects?: () => void;
  onAbout?: () => void;
  currentUser?: CurrentUser | null;
  onLogout?: () => void;
  onHome?: () => void;
  onDocs?: () => void;
  onAccount?: () => void;
}

export default function Navbar({
  onLogin,
  onGetStarted,
  onMyProjects,
  onAbout,
  currentUser,
  onLogout,
  onHome,
  onDocs,
  onAccount,
}: NavbarProps) {
  const [showDropdown, setShowDropdown] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-[#F5F7F7] border-b border-gray-200 h-14 flex items-center px-6">
      <button
        onClick={onHome}
        className="text-[#1D3335] font-large text-lg mr-8 hover:opacity-70 transition-opacity cursor-pointer"
      >
        mothra
      </button>
      <div className="hidden sm:flex items-center gap-6 flex-1">
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            onAbout?.();
          }}
          className="text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
        >
          about mothra
        </a>
        <a
          href="#"
          onClick={(e) => { e.preventDefault(); onDocs?.(); }}
          className="text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
        >
          documentation / walkthrough
        </a>
      </div>
      <div className="flex items-center gap-4 ml-auto">
        <a
          href="https://github.com/DDMAL/mothra"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[#1D3335] hover:opacity-70 transition-opacity"
          aria-label="GitHub repository"
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M12 0C5.373 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.6.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.51 11.51 0 0 1 12 6.803c.93.004 1.867.125 2.747.368 2.29-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
          </svg>
        </a>
        <button
          onClick={onMyProjects}
          className="text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
        >
          my projects
        </button>

        {currentUser ? (
          <div className="relative">
            <button
              onClick={() => setShowDropdown((v) => !v)}
              className="px-5 py-2 bg-[#4AADAA] text-white text-sm rounded-full hover:opacity-90 transition-opacity cursor-pointer"
            >
              hello, {currentUser.username}!
            </button>
            {showDropdown && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  onClick={() => setShowDropdown(false)}
                />
                <div className="absolute right-0 top-full mt-2 z-50 bg-white border border-gray-200 rounded-2xl shadow-lg py-2 min-w-[160px]">
                  <button
                    onClick={() => { setShowDropdown(false); onAccount?.(); }}
                    className="w-full text-left px-5 py-2.5 text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
                  >
                    my account
                  </button>
                  <button
                    onClick={() => {
                      setShowDropdown(false);
                      onLogout?.();
                    }}
                    className="w-full text-left px-5 py-2.5 text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
                  >
                    log out
                  </button>
                </div>
              </>
            )}
          </div>
        ) : (
          <>
            <button
              onClick={onLogin}
              className="text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
            >
              log in
            </button>
            <button
              onClick={onGetStarted}
              className="px-5 py-2 bg-[#4AADAA] text-white text-sm rounded-full hover:opacity-90 transition-opacity cursor-pointer"
            >
              get started
            </button>
          </>
        )}
      </div>
    </nav>
  );
}
