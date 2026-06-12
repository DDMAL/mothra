import { useState } from "react";
import type { CurrentUser } from "../hooks/useAuth";

type AccountTab = "info" | "files";

interface MyAccountProps {
    currentUser: CurrentUser;
}

export default function MyAccount({ currentUser }: MyAccountProps) {
    const [tab, setTab] = useState<AccountTab>("info");

    return (
        <div className="animate-fade-in flex-1 flex bg-[#4AADAA]">
            <aside className="w-72 shrink-0 bg-[#c5dfe0] flex flex-col py-6 px-2
            sticky top-14 self-start h-[calc(100vh-3.5rem)] overflow-y-auto">
                <nav className="flex flex-col gap-0.5 text-sm text-[#1D3335]">
                    <button
                        onClick={() => setTab("info")}
                        className={`text-left font-bold px-4 py-1.5 rounded-lg transition-colors ${
                            tab === "info" ? "bg-[#4AADAA]/40" : "hover:bg-[#4AADAA]/20"
                        }`}
                    >
                        account information
                    </button>
                    <button
                        onClick={() => setTab("files")}
                        className={`text-left font-bold px-4 py-1.5 rounded-lg mt-2 transition-colors ${
                            tab === "files" ? "bg-[#4AADAA]/40" : "hover:bg-[#4AADAA]/20"
                        }`}
                    >
                        files
                    </button>
                </nav>
            </aside>

            <main className="flex-1 overflow-y-auto px-10 py-8">
                <h1 className="text-3xl font-bold italic text-[#1D3335] mb-6">my account</h1>
                {tab === "info" && (
                    <div className="bg-[#C8E6E3] rounded-3xl p-8 max-w-2xl flex flex-col gap-5">
                        <div className="flex items-center gap-4">
                            <span className="text-[#1D3335] w-24">username:</span>
                            <input
                                type="text"
                                readOnly
                                defaultValue={currentUser.username}
                                className="flex-1 bg-white rounded-2xl px-4 py-2 text-[#1D3335] outline-none"
                            />
                            <button
                                onClick={() => {}}
                                className="px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                            >
                                change
                            </button>
                        </div>
                        <div className="flex items-center gap-4">
                            <span className="text-[#1D3335] w-24">email:</span>
                            <input
                                type="text"
                                readOnly
                                defaultValue={currentUser.email}
                                className="flex-1 bg-white rounded-2xl px-4 py-2 text-[#1D3335] outline-none"
                            />
                            <button
                                onClick={() => {}}
                                className="px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                            >
                                change
                            </button>
                        </div>
                        <p className="text-[#1D3335]">account created: —</p>
                        <button
                            onClick={() => {}}
                            className="self-start px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                        >
                            change password
                        </button>
                        <button
                            onClick={() => {}}
                            className="self-start px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                        >
                            delete my account
                        </button>
                    </div>
                )}
            </main>
        </div>
    );
}