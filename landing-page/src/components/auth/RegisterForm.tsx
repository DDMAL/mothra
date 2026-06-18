import { useState } from "react";
import type { CurrentUser } from "../hooks/useAuth";

interface RegisterFormProps {
  onSwitchToLogin: () => void;
  onSuccess: (user: CurrentUser, token: string) => void;
}


export default function RegisterForm({ onSwitchToLogin, onSuccess }: RegisterFormProps) {
    const [firstName, setFirstName] = useState("");
    const [lastName, setLastName] = useState("");
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [error, setError] = useState("");

    const handleSubmit = async() => {
      setError("");
    if (password !== confirmPassword)  { setError("passwords do not match"); return; }
    const r = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, first_name: firstName, last_name: lastName, password, }),
    });
    if (!r.ok) {
      const data = await r.json().catch(() => ({}));
      setError((data as { detail?: string }).detail || "registration failed");
      return;
    }
    const { token, user } = await r.json();
    onSuccess(user, token);
  };

  return (
    <div className="bg-[#C8E6E3] rounded-3xl p-6 sm:p-10 flex flex-col gap-4">
      <h2 className="text-xl text-[#1D3335] text-center mb-2">
        new to mothra? create an account:
      </h2>
      <div className="flex gap-3">
        <input
          type="text"
          placeholder="first name"
          value={firstName}
          onChange={(e) => setFirstName(e.target.value)}
          className="flex-1 bg-white rounded-2xl px-4 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
        />
        <input
          type="text"
          placeholder="last name"
          value={lastName}
          onChange={(e) => setLastName(e.target.value)}
          className="flex-1 bg-white rounded-2xl px-4 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
        />
      </div>
      <input
        type="text"
        placeholder="username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <input
        type="email"
        placeholder="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <input
        type="password"
        placeholder="enter password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <input
        type="password"
        placeholder="confirm password"
        value={confirmPassword}
        onChange={(e) => setConfirmPassword(e.target.value)}
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      {error && <p className="text-red-600 text-xs text-center">{error}</p>}
      <button
        onClick={handleSubmit}
        className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
      >
        register
      </button>
      <hr className="border-[#1D3335]/20" />
      <button
        onClick={onSwitchToLogin}
        className="text-sm text-[#1D3335] text-center hover:opacity-70 transition-opacity cursor-pointer"
      >
        have an account already? login here
      </button>
    </div>
  );
}
