import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";
import type { AuthSession } from "../../hooks/useAuth";

interface AuthPageProps {
  mode: "login" | "register";
  onSwitchMode: (mode: "login" | "register") => void;
  onSuccess: (session: AuthSession) => void;
  // Why the form is showing, when the user didn't ask for it -- e.g. AppRouter
  // standing it in for a page that needs a logged-in user.
  notice?: string;
}

export default function AuthPage({
  mode,
  onSwitchMode,
  onSuccess,
  notice,
}: AuthPageProps) {
  return (
    <div className="flex-1 bg-[#4AADAA] flex items-center justify-center px-6 py-16">
      <div key={mode} className="animate-fade-in w-full max-w-lg">
        {notice && (
          <p
            role="status"
            className="mb-4 text-center text-sm font-semibold text-[#1D3335]"
          >
            {notice}
          </p>
        )}
        {mode === "login" ? (
          <LoginForm
            onSwitchToRegister={() => onSwitchMode("register")}
            onSuccess={onSuccess}
          />
        ) : (
          <RegisterForm
            onSwitchToLogin={() => onSwitchMode("login")}
            onSuccess={onSuccess}
          />
        )}
      </div>
    </div>
  );
}
