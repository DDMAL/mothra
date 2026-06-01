import { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Features from './components/Features';
import Footer from'./components/Footer';
import AuthPage from './components/AuthPage';

type View = 'landing' | 'login' | 'register';

export default function App() {
    const [view, setView] = useState<View>('landing');

    useEffect(() => {
        if (view !== 'landing') return;

        const heroTargets = document.querySelectorAll('.hero-fade');
        const timer = setTimeout(() => {
            heroTargets.forEach((el) => el.classList.add('visible'));
        }, 100);

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                        observer.unobserve(entry.target);
                    }
                });
            },
            {threshold: 0.1},
        );

        document.querySelectorAll('.scroll-fade').forEach((el) => observer.observe(el));

        return () => {
            clearTimeout(timer);
            observer.disconnect();
        };
    }, [view]);

    return (
        <div className="min-h-screen flex flex-col">
            <Navbar 
                onLogin={() => setView('login')}
                onGetStarted={() => setView('register')}
            />
            {view === 'landing' ? (
                <main>
                    <Hero />
                    <Features />
                </main>
            ) : (
                <AuthPage mode={view} onSwitchMode={(m) => setView(m)} />
            )}
            <Footer />
        </div>
    );
}