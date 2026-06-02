import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";

interface AuthPageProps {
  mode: "login" | "register";
  onSwitchMode: (mode: "login" | "register") => void;
}

export default function AuthPage({ mode, onSwitchMode }: AuthPageProps) {
  return (
    <div className="flex-1 bg-[#4AADAA] flex items-center justify-center px-6 py-16">
      <div key={mode} className="animate-fade-in w-full max-w-lg">
        {mode === "login" ? (
          <LoginForm onSwitchToRegister={() => onSwitchMode("register")} />
        ) : (
          <RegisterForm onSwitchToLogin={() => onSwitchMode("login")} />
        )}
      </div>
    </div>
  );
}
