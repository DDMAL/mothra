import LoginForm from "./LoginForm";
import RegisterForm from "../RegisterForm";
import type { CurrentUser } from "../../hooks/useAuth";

interface AuthPageProps {
  mode: "login" | "register";
  onSwitchMode: (mode: "login" | "register") => void;
  onSuccess: (user: CurrentUser, token: string) =>  void;
}

export default function AuthPage({ mode, onSwitchMode, onSuccess }: AuthPageProps) {
  return (
    <div className="flex-1 bg-[#4AADAA] flex items-center justify-center px-6 py-16">
      <div key={mode} className="animate-fade-in w-full max-w-lg">
        {mode === "login" ? (
          <LoginForm onSwitchToRegister={() => onSwitchMode("register")} onSuccess={onSuccess} />
        ) : (
          <RegisterForm onSwitchToLogin={() => onSwitchMode("login")} onSuccess={onSuccess} />
        )}
      </div>
    </div>
  );
}
