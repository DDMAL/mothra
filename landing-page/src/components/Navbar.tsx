interface NavbarProps {
    onLogin?: () => void;
    onGetStarted?: () => void;
}

export default function Navbar({ onLogin, onGetStarted }: NavbarProps) {
    return (
        <nav className="sticky top-0 z-50 bg-[#F5F7F7] border-b border-gray-200 h-14 flex items-center px-6">
            <span className="text-[#1D3335] font-medium text-lg mr-8">mothra</span>
            <div className="flex items-center gap-6 flex-1">
                <a
                    href="#"
                    onClick={(e) => e.preventDefault()}
                    className="text=sm text-[#1D3335] hover:opacity-70 transition-opacity"
                >
                    about mothra
                </a>
                <a
                    href="#"
                    onClick={(e) => e.preventDefault()}
                    className="text=sm text-[#1D3335] hover:opacity-70 transition-opacity"
                >
                    documentation / walkthrough
                </a>
            </div>
            <div className="flex items-center gap-4">
                <button
                    onClick={onLogin}
                    className="text-sm text-[#1D3335] hover:opacity-70 transition-opacity"
                >
                    log in
                </button>
                <button
                    onClick={onGetStarted}
                    className="px-5 py-2 bg-[#4AADAA] text-white text-sm rounded-full hover:opacity-90 transition-opacity"
                >
                    get started
                </button>
            </div>
        </nav>
    );
}