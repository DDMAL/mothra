interface LoginFormProps {
  onSwitchToRegister: () => void;
}

export default function LoginForm({ onSwitchToRegister }: LoginFormProps) {
  return (
    <div className="bg-[#C8E6E3] rounded-3xl p-6 sm:p-10 flex flex-col gap-4">
      <h2 className="text-2xl text-[#1D3335] text-center mb-2">log in</h2>
      <input
        type="text"
        placeholder="username/email"
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <input
        type="password"
        placeholder="password"
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <button
        onClick={(e) => e.preventDefault()}
        className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
      >
        log in
      </button>
      <hr className="border-[#1D3335]/20" />
      <button
        onClick={onSwitchToRegister}
        className="text-sm text-[#1D3335] text-center hover:opacity-70 transition-opacity cursor-pointer"
      >
        new user? register for an account
      </button>
    </div>
  );
}
