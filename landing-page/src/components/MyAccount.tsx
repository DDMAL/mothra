import { useState, useRef, useEffect } from "react";
import type { CurrentUser } from "../hooks/useAuth";
import { authHeaders } from "../hooks/useAuth";

type AccountTab = "info" | "files";

interface MyAccountProps {
    currentUser: CurrentUser;
    onUserUpdate: (u: CurrentUser) => void;
}

function formatDate(s: string) {
    const d = new Date(s.replace(" ", "T") + "Z");
    return d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}

export default function MyAccount({ currentUser, onUserUpdate}: MyAccountProps) {
    const [tab, setTab] = useState<AccountTab>("info");

    const [editingUsername, setEditingUsername] = useState(false);
    const [editingEmail, setEditingEmail] = useState(false);
    const [usernameVal, setUsernameVal] = useState(currentUser.username);
    const [emailVal, setEmailVal] = useState(currentUser.email);
    const [usernameError, setUsernameError] = useState("");
    const [emailError, setEmailError] = useState("");

    const [pendingField, setPendingField] = useState<"username" | "email" | null>(null);
    const [pendingValue, setPendingValue] = useState("");

    const usernameRef = useRef<HTMLInputElement>(null);
    const emailRef = useRef<HTMLInputElement>(null);
    const confirmRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        if (editingUsername) usernameRef.current?.focus();
    }, [editingUsername]);

    useEffect(() => {
        if (!pendingField) return;
        const t = setTimeout(() => confirmRef.current?.focus(), 50);
        return () => clearTimeout(t);
    }, [pendingField]);
    useEffect(() => {
        if (editingEmail) emailRef.current?.focus();
    }, [editingEmail]);

    function handleUsernameKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key === "Escape") {
            setUsernameVal(currentUser.username);
            setEditingUsername(false);
            setUsernameError("");
        }
        if (e.key === "Enter") {
            const v = usernameVal.trim();
            if (!v || v === currentUser.username) {
                setUsernameVal(currentUser.username);
                setEditingUsername(false);
                return;
            }
            setPendingField("username");
            setPendingValue(v);
        }
    }

    function handleEmailKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key === "Escape") {
            setEmailVal(currentUser.email);
            setEditingEmail(false);
            setEmailError("");
        }
        if (e.key === "Enter") {
            const v = emailVal.trim();
            if (!v || v === currentUser.email) {
                setEmailVal(currentUser.email);
                setEditingEmail(false);
                return;
            }
            setPendingField("email");
            setPendingValue(v);
        }
    }

    async function handleConfirm() {
        const body = pendingField === "username" ? { username: pendingValue } : { email: pendingValue };
         const res = await fetch("/api/me", {
            method: "PATCH",
            headers: { ...authHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify(body),
         });
         if (res.status === 409) {
            if (pendingField === "username") setUsernameError("username already taken");
            else setEmailError("email already in use");
            setPendingField(null);
            return;
         }
         if (res.ok) {
            const updated = await res.json();
            onUserUpdate(updated);
            if (pendingField === "username") {
                setUsernameVal(updated.username);
                setEditingUsername(false);
                setUsernameError("");
            } else {
                setEmailVal(updated.email);
                setEditingEmail(false);
                setEmailError("");
            }
         }
         setPendingField(null);
    }

    function handleCancelConfirm() {
        if (pendingField === "username") setUsernameVal(currentUser.username);
        else setEmailVal(currentUser.email);
        setPendingField(null);
    }

    const previousValue = pendingField === "username" ? currentUser.username : currentUser.email;

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
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-4">
                                <span className="text-[#1D3335] w-24">username:</span>
                                <input
                                    ref={usernameRef}
                                    type="text"
                                    readOnly={!editingUsername}
                                    value={usernameVal}
                                    onChange={(e) => setUsernameVal(e.target.value)}
                                    onKeyDown={handleUsernameKeyDown}
                                    className={`flex-1 bg-white rounded-2xl px-4 py-2 text-[#1D3335] outline-none transition-shadow ${editingUsername ? "ring-2 ring-[#1E6B70]" : ""}`}
                                />
                                <button
                                    onClick={() => { setEditingUsername(true); setUsernameError(""); }}
                                    className="px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                                >
                                    change
                                </button>
                            </div>
                            {usernameError && <p className="text-red-600 text-sm pl-28">{usernameError}</p>}
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-4">
                                <span className="text-[#1D3335] w-24">email:</span>
                                <input
                                    ref={emailRef}
                                    type="text"
                                    readOnly={!editingEmail}
                                    value={emailVal}
                                    onChange={(e) => setEmailVal(e.target.value)}
                                    onKeyDown={handleEmailKeyDown}
                                    className={`flex-1 bg-white rounded-2xl px-4 py-2 text-[#1D3335] outline-none transition-shadow ${editingEmail ? "ring-2 ring-[#1E6B70]" : ""}`}
                                />
                                <button
                                    onClick={() => { setEditingEmail(true); setEmailError(""); }}
                                    className="px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                                >
                                    change
                                </button>
                            </div>
                            {emailError && <p className="text-red-600 text-sm pl-28">{emailError}</p>}
                        </div>
                        <p className="text-[#1D3335]">account created: {currentUser.createdAt ? formatDate(currentUser.createdAt) : "—"}</p>
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

            {pendingField && (
                <>
                    <div className="fixed inset-0 z-40 bg-black/30" onClick={handleCancelConfirm} />
                    <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-sm bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-4 shadow-2xl">
                        <p className="text-[#1D3335] text-center">
                            are you sure you want to change your {pendingField} from{" "}
                            <strong>"{previousValue}"</strong> to{" "}
                            <strong>"{pendingValue}"</strong>?
                        </p>
                        <div className="flex gap-3 justify-center">
                            <button
                                ref={confirmRef}
                                onClick={handleConfirm}
                                className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold hover:opacity-90 transition-opacity cursor-pointer"
                            >
                                confirm
                            </button>
                            <button
                                onClick={handleCancelConfirm}
                                className="bg-white text-[#1D3335] rounded-xl px-6 py-3 text-sm font-bold hover:opacity-80 transition-opacity cursor-pointer"
                            >
                                cancel
                            </button>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}