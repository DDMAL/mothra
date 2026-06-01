interface RegisterFormProps {
    onSwitchToLogin: () => void;
}

export default function RegisterForm({ onSwitchToLogin }: RegisterFormProps) {
    return (
    <div className="bg-[#C8E6E3] rounded-3xl p-6 sm:p-10 flex flex-col gap-4">
      <h2 className="text-xl text-[#1D3335] text-center mb-2">
        new to mothra? create an account:
      </h2>
      <div className="flex gap-3">
        <input
          type="text"
          placeholder="first name"
          className="flex-1 bg-white rounded-2xl px-4 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
        />
        <input
          type="text"
          placeholder="last name"
          className="flex-1 bg-white rounded-2xl px-4 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
        />
      </div>
      <input
        type="text"
        placeholder="username"
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <input
        type="email"
        placeholder="email"
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <input
        type="password"
        placeholder="enter password"
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <input
        type="password"
        placeholder="confirm password"
        className="w-full bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
      />
      <button
        onClick={(e) => e.preventDefault()}
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