import { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Features from './components/Features';
import Footer from'./components/Footer';
import AuthPage from './components/AuthPage';
import MyProjects from './components/MyProjects';
import ProjectDetail from './components/ProjectDetail';

type View = 'landing' | 'login' | 'register' | 'projects' | 'project';

export interface ProjectImage { id: string; name: string; }
export interface Project { name: string; user: string; images: ProjectImage[]; }

export default function App() {
    const [view, setView] = useState<View>('landing');
    const [projects, setProjects] = useState<Project[]>([
    { name: 'project alpha', user: 'username', images: [
        {id: '1', name: 'image 1'}, { id: '2', name: 'image 2'}, {id: '3', name: 'image 3' },
    ]},
    { name: 'project beta', user: 'username', images: Array.from({ length: 7 }, (_, i) => ({ id: String(i + 1), name: `image ${i + 1}` })) },
  ]);
    const [selectedProject, setSelectedProject] = useState<string | null>(null);

    useEffect(() => {
        if (view !== 'landing') {
            document.querySelectorAll('.fade-target').forEach((el) => el.classList.add('visible'));
            return;
        }

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
                onMyProjects={() => setView('projects')}
                loggedIn={view === 'projects' || view === 'project'}
                onHome={() => setView('landing')}
                onLogout={() => setView('landing')}
            />
            {view === 'landing' ? (
                <main>
                    <Hero 
                        onGetStarted={() => setView('register')}
                    />
                    <Features />
                </main>
            ) : view === 'projects' ? (
                <MyProjects 
                    projects={projects}
                    setProjects={setProjects}
                    onSelectProject={(name) => { setSelectedProject(name); setView('project'); }}
                />
            ) : view === 'project' && selectedProject ? (
                <ProjectDetail
                    project={projects.find((p) => p.name === selectedProject)!}
                    onBack={() => setView('projects')}
                    onUpdateProject={(updated) => 
                        setProjects((prev) => prev.map((p) => (p.name === updated.name ? updated : p)))
                    } 
                />
            ) :
            (
                <AuthPage mode={view as 'login' | 'register'} onSwitchMode={(m) => setView(m)} />
            )}
            <Footer />
        </div>
    );
}