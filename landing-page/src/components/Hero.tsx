export default function Hero() {
    return (
        <section className="bg-[#4AADAA] py-16">
            <div className="max-w-4xl mx-auto text-center">
                <h1 className="hero-fade fade-target text-4xl font-bold italic text-[#1D3335] mb-3">
                    welcome to mothra!
                </h1>
                <p className="hero-fade fade-target text-lg text-[#1D3335] mb-12">
                    a project created by the DDMAL Lab at McGill University
                </p>
                <div className="hero-fade fade-target bg-[#1E6B70] rounded-lg h-64 flex items-center justify-center">
                    <span className="text-white/60 text-sm">image of interface</span>
                </div>
            </div>
        </section>
    );
}