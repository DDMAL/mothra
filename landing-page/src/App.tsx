import { useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Features from './components/Features';
import Footer from'./components/Footer';

export default function App() {
    useEffect(() => {
        const heroTargets = document.querySelectorAll('.hero-fade');
        setTimeout(() => {
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

        return () => observer.disconnect();
    }, []);

    return (
        <div className="min-h-screen flex flex-col">
            <Navbar />
            <main>
                <Hero />
                <Features />
            </main>
            <Footer />
        </div>
    );
}