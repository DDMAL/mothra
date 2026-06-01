import { useState } from 'react';

interface NavbarProps {
    onLogin?: () => void;
    onGetStarted?: () => void;
    onMyProjects?: () => void;
    loggedIn?: boolean;
    onLogout?: () => void;
}

export default function Navbar({ onLogin, onGetStarted, onMyProjects, loggedIn, onLogout }: NavbarProps) {
    const [showDropdown, setShowDropdown] = useState(false);

    return (
        <nav className="sticky top-0 z-50 bg-[#F5F7F7] border-b border-gray-200 h-14 flex items-center px-6">
            <span className="text-[#1D3335] font-large text-lg mr-8">mothra</span>
            <div className="hidden sm:flex items-center gap-6 flex-1">
                <a
                    href="#"
                    onClick={(e) => e.preventDefault()}
                    className="text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
                >
                    about mothra
                </a>
                <a
                    href="#"
                    onClick={(e) => e.preventDefault()}
                    className="text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
                >
                    documentation / walkthrough
                </a>
            </div>
            <div className="flex items-center gap-4 ml-auto">
                <button
                    onClick={onMyProjects}
                    className="text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
                >
                    my projects
                </button>

                {loggedIn ? (
                    <div className="relative">
                        <button
                            onClick={() => setShowDropdown((v) => !v)}
                            className="px-5 py-2 bg-[#4AADAA] text-white text-sm rounded-full hover:opacity-90 transition-opacity cursor-pointer"
                        >
                            hello, [name]!
                        </button>
                        {showDropdown && (
                            <>
                                <div
                                    className="fixed inset-0 z-40"
                                    onClick={() => setShowDropdown(false)}
                                />
                                <div className="absolute right-0 top-full mt-2 z-50 bg-white border border-gray-200 rounded-2xl shadow-lg py-2 min-w-[160px]">
                                    <button
                                        onClick={() => setShowDropdown(false)}
                                        className="w-full text-left px-5 py-2.5 text-sm text-[#1D3335] hover:opacity-70 transition-opacity cursor-pointer"
                                    >
                                        my account
                                    </button>
                                    <button
                                        onClick={() => { setShowDropdown(false); onLogout?.(); }}
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