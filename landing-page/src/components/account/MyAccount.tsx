import React, { useState, useRef, useEffect } from "react";
import type { CurrentUser } from "../../hooks/useAuth";
import { authHeaders } from "../../hooks/useAuth";
import type { Project } from "../../types";
import Modal from "../shared/Modal";

type AccountTab = "info" | "files";

type UsageData = {
    projects: {total: number; active: number; deleted: number; };
    images: { count: number; bytes: number; };
    meiFiles: { count: number; bytes: number; corrected: number; };
    logs: { count: number; bytes: number; };
    quotaBytes: number;
};

interface MyAccountProps {
    currentUser: CurrentUser;
    onUserUpdate: (u: CurrentUser) => void;
    onLogout: () => void;
}


function formatDate(s: string) {
    const d = new Date(s.replace(" ", "T") + "Z");
    return d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}

function formatBytes(n: number): string {
    if (n === 0) return "0 B";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function MyAccount({ currentUser, onUserUpdate, onLogout}: MyAccountProps) {
    const [tab, setTab] = useState<AccountTab>("info");

    const [editingUsername, setEditingUsername] = useState(false);
    const [editingEmail, setEditingEmail] = useState(false);
    const [usernameVal, setUsernameVal] = useState(currentUser.username);
    const [emailVal, setEmailVal] = useState(currentUser.email);
    const [usernameError, setUsernameError] = useState("");
    const [emailError, setEmailError] = useState("");

    const [pendingField, setPendingField] = useState<"username" | "email" | null>(null);
    const [pendingValue, setPendingValue] = useState("");

    const [showChangePassword, setShowChangePassword] = useState(false);
    const [showDeleteAccount, setShowDeleteAccount] = useState(false);

    // change password form
    const [oldPw, setOldPw] = useState("");
    const [newPw, setNewPw] = useState("");
    const [confirmPw, setConfirmPw] = useState("");
    const [pwError, setPwError] = useState("");

    // delete account
    const [deleteCounts, setDeleteCounts] = useState<{projects:number, images:number, mei:number} | null>(null);
   
    const usernameRef = useRef<HTMLInputElement>(null);
    const emailRef = useRef<HTMLInputElement>(null);
    const confirmRef = useRef<HTMLButtonElement>(null);

    const [usage, setUsage] = useState<UsageData | null>(null);
    const usedBytes = (usage?.images.bytes ?? 0) + (usage?.meiFiles.bytes ?? 0) + (usage?.logs.bytes ?? 0);
    const pct = usage?.quotaBytes ? Math.min((usedBytes / usage.quotaBytes) * 100, 100) : 0;

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

    useEffect(() => {
    if (tab !== "files") return;
    fetch("/api/me/usage", { headers: authHeaders() })
        .then(r => r.json())
        .then(setUsage);
    }, [tab]);

    useEffect(() => {
        if (!showDeleteAccount) return;
        fetch("/api/projects", { headers: authHeaders() })
            .then(r => r.json())
            .then((projects: Project[]) => {
                const active = projects.filter(p => !p.deletedAt);
                setDeleteCounts({
                    projects: active.length,
                    images: active.reduce((n, p) => n + p.images.length, 0),
                    mei: active.reduce((n, p) => n + p.meiFiles.length, 0),
                });
            });
    }, [showDeleteAccount]);

    // changing username or email

    const handleFieldKeyDown= (field: "username" | "email") => (e: React.KeyboardEvent<HTMLInputElement>) => {
        const isUsername = field === "username";
        const val = isUsername ? usernameVal : emailVal;
        const stored = isUsername  ? currentUser.username : currentUser.email;
        const setVal = isUsername ? setUsernameVal : setEmailVal; 
        const setEditing = isUsername ? setEditingUsername : setEditingEmail;
        const setError = isUsername ? setUsernameError : setEmailError;

        if (e.key === "Escape") {
            setVal(stored);
            setEditing(false);
            setError("");
        }
        if (e.key === "Enter") {
            const v = val.trim();
            if (!v || v === stored) {
                setVal(stored); setEditing(false); setError(""); return;
            }
            setPendingField(field);
            setPendingValue(v);
        }
    };

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

    function closePasswordModal() {
        setShowChangePassword(false);
        setOldPw("");
        setNewPw(""); 
        setConfirmPw(""); 
        setPwError("");
    }
    const previousValue = pendingField === "username" ? currentUser.username : currentUser.email;

    return (
        <>
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
                        <EditableField
                            label="username"
                            value={usernameVal}
                            onChange={setUsernameVal}
                            onKeyDown={handleFieldKeyDown("username")}
                            isEditing={editingUsername}
                            onEdit={() => { setEditingUsername(true); setUsernameError(""); }}
                            error={usernameError}
                            inputRef={usernameRef}
                        />
                        <EditableField
                            label="email"
                            value={emailVal}
                            onChange={setEmailVal}
                            onKeyDown={handleFieldKeyDown("email")}
                            isEditing={editingEmail}
                            onEdit={() => { setEditingEmail(true); setEmailError(""); }}
                            error={emailError}
                            inputRef={emailRef}
                        />
                        <p className="text-[#1D3335]">account created: {currentUser.createdAt ? formatDate(currentUser.createdAt) : "—"}</p>
                        <div className="flex gap-3">
                            <button
                            onClick={() => setShowChangePassword(true)}
                            className="self-start px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                            >
                                change password
                            </button>
                            <button
                                onClick={() => setShowDeleteAccount(true)}
                                className="self-start px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                            >
                                delete my account
                            </button>
                        </div>
                    </div>
                )}
                {tab === "files" && (
                    <div className="flex flex-col gap-6 max-w-2xl">
                        {!usage ? (
                            <p className="tet-white/60 text-sm">loading...</p>
                        ) : (
                            <>
                                <div className="bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-2">
                                    <p className="text-[#1D3335] font-semibold text-lg mb-1">total storage</p>
                                    <p className="text-4xl font-bold text-[#1D3335]">
                                        {formatBytes(usage.images.bytes + usage.meiFiles.bytes + usage.logs.bytes)}
                                    </p>
                                    <div className="...">
                                        <div className="flex justify-between text-xs text-[#1D3335]/70 mb-1">
                                            <span>{formatBytes(usedBytes)} used</span>
                                            <span>{formatBytes(usage.quotaBytes)} limit</span>
                                        </div>
                                        <div className="w-full bg-white/30 rounded-full h-2 overflow-hidden">
                                            <div
                                            className="h-full rounded-full transition-all"
                                            style={{
                                                width: `${pct}%`,
                                                background: pct > 90 ? "#dc2626" : "#1E6B70",
                                            }}
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-[#C8E6E3] rounded-3xl p-6 flex flex-col gap-1">
                                        <p className="text-[#1D3335]/60 text-xs uppercase tracking-wide">projects</p>
                                        <p className="text-3xl font-bold text-[#1D3335]">{usage.projects.active}</p>
                                        <p className="text-sm text-[#1D3335]/70">
                                            {usage.projects.deleted > 0 ? `+ ${usage.projects.deleted} in trash` : "none in trash"}
                                        </p>
                                    </div>

                                    <div className="bg-[#C8E6E3] rounded-3xl p-6 flex flex-col gap-1">
                                        <p className="text-[#1D3335]/60 text-xs uppercase tracking-wide">images</p>
                                        <p className="text-3xl font-bold text-[#1D3335]">{usage.images.count}</p>
                                        <p className="text-sm text-[#1D3335]/70">{formatBytes(usage.images.bytes)}</p>
                                    </div>

                                    <div className="bg-[#C8E6E3] rounded-3xl p-6 flex flex-col gap-1">
                                        <p className="text-[#1D3335]/60 text-xs uppercase tracking-wide">mei files</p>
                                        <p className="text-3xl font-bold text-[#1D3335]">{usage.meiFiles.count}</p>
                                        <p className="text-sm text-[#1D3335]/70">
                                            {usage.meiFiles.corrected}/{usage.meiFiles.count} corrected
                                            &nbsp;·&nbsp;{formatBytes(usage.meiFiles.bytes)}
                                        </p>
                                    </div>

                                    <div className="bg-[#C8E6E3] rounded-3xl p-6 flex flex-col gap-1">
                                        <p className="text-[#1D3335]/60 text-xs uppercase tracking-wide">encoding logs</p>
                                        <p className="text-3xl font-bold text-[#1D3335]">{usage.logs.count}</p>
                                        <p className="text-sm text-[#1D3335]/70">{formatBytes(usage.logs.bytes)}</p>
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                )}
            </main>
        </div>

        {pendingField && (
           <Modal size="sm" backdrop="dim" onClose={handleCancelConfirm} showCloseButton={false}>
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
           </Modal>    
        )}

        {showChangePassword && (
            <Modal size="sm" backdrop="dim" onClose={closePasswordModal}>
                <h2 className="text-xl text-[#1D3335] text-center">change password</h2>
                    <input autoFocus type="password" placeholder="old password" value={oldPw} onChange={e => setOldPw(e.target.value)}
                        className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60" />
                    <input type="password" placeholder="new password" value={newPw} onChange={e => setNewPw(e.target.value)}
                        className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60" />
                    <input type="password" placeholder="confirm new password" value={confirmPw} onChange={e => setConfirmPw(e.target.value)}
                        className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60" />
                    {pwError && <p className="text-red-600 text-xs text-center">{pwError}</p>}
                    <button
                        onClick={async () => {
                            setPwError("");
                            if (newPw !== confirmPw) { setPwError("passwords do not match"); return; }
                            const res = await fetch("/api/me/password", {
                                method: "PATCH",
                                headers: { ...authHeaders(), "Content-Type": "application/json" },
                                body: JSON.stringify({ old_password: oldPw, new_password: newPw }),
                            });
                            if (!res.ok) {
                                const d = await res.json().catch(() => ({}));
                                setPwError((d as { detail?: string }).detail || "failed to change password");
                                return;
                            }
                            setShowChangePassword(false); setOldPw(""); setNewPw(""); setConfirmPw(""); setPwError("");
                        }}
                        className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
                    >change password</button>
            </Modal>   
        )}

        {showDeleteAccount && (
            <Modal size="sm" backdrop="dim" onClose={() => setShowDeleteAccount(false)}>
                    <h2 className="text-xl text-[#1D3335] text-center">confirm account deletion</h2>
                    <p className="text-sm text-[#1D3335] text-center leading-relaxed">
                        are you sure you want to delete your account?<br />this is irreversible and you will lose:
                    </p>
                    <ul className="text-sm text-[#1D3335] list-disc list-inside">
                        <li>{deleteCounts ? deleteCounts.projects : "…"} projects</li>
                        <li>{deleteCounts ? deleteCounts.images : "…"} images</li>
                        <li>{deleteCounts ? deleteCounts.mei : "…"} mei files</li>
                    </ul>
                    <div className="flex gap-3 justify-center">
                        <button
                            onClick={async () => {
                                await fetch("/api/me", { method: "DELETE", headers: authHeaders() });
                                onLogout();
                            }}
                            className="px-6 py-2.5 bg-[#1E6B70] text-white font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
                        >yes, delete account</button>
                        <button
                            onClick={() => setShowDeleteAccount(false)}
                            className="px-6 py-2.5 border-2 border-[#1D3335]/30 text-[#1D3335] font-semibold rounded-xl hover:opacity-70 cursor-pointer text-sm"
                        >cancel</button>
                    </div>
            </Modal>
        )}
        </>
    );
}

function EditableField({ label, value, onChange, onKeyDown, isEditing, onEdit, error, inputRef } : {
    label: string;
    value: string;
    onChange: (v: string) => void;
    onKeyDown: React.KeyboardEventHandler<HTMLInputElement>;
    isEditing: boolean;
    onEdit: () => void;
    error: string;
    inputRef: React.RefObject<HTMLInputElement | null>;
}) {
    return (
        <div className="flex flex-col gap-1">
            <div className="flex items-center gap-4">
                <span className="text-[#1D3335] w-24">{label}:</span>
                <input
                    ref={inputRef}
                    type="text"
                    readOnly={!isEditing}
                    value={value}
                    onChange={e => onChange(e.target.value)}
                    onKeyDown={onKeyDown}
                    className={`flex-1 bg-white rounded-2xl px-4 py-2 text-[#1D3335] outline-none transition-shadow ${isEditing ? "ring-2 ring-[#1E6B70]" : ""}`}
                />
                <button
                    onClick={onEdit}
                    className="px-5 py-2 bg-[#1E6B70] text-white text-sm rounded-2xl hover:opacity-90 transition-opacity cursor-pointer"
                >
                    change
                </button>
            </div>
            {error && <p className="text-red-600 text-sm pl-28">{error}</p>}
        </div>
    );
}