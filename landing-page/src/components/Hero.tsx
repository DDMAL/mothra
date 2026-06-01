export default function Hero() {
    return (
        <section className="bg-[#4AADAA] py-16 px-4">
            <div className="max-w-4xl mx-auto text-center">
                <h1 className="hero-fade fade-target text-3xl sm:text-4xl font-bold italic text-white mb-3">
                    welcome to mothra!
                </h1>
                <p className="hero-fade fade-target text-lg text-[#1D3335] mb-8">
                    a project created by the DDMAL Lab at McGill University
                </p>
                <div className="hero-fade fade-target flex flex-wrap items-center justify-center gap-4 mb-12">
                    <button
                        onClick={(e) => e.preventDefault()}
                        className="px-6 py-2.5 bg-white text-[#1D3335] text-sm rounded-full hover:opacity-90 transition-opacity cursor-pointer"
                    >
                        view walkthrough
                    </button>
                    <button
                        onClick={(e) => e.preventDefault()}
                        className="px-6 py-2.5 bg-[#4AADAA] text-white text-sm rounded-full border border-white hover:opacity-90 transition-opacity cursor-pointer"
                    >
                        get started now
                    </button>
                </div>
                <div className="hero-fade fade-target bg-[#1E6B70] rounded-lg h-64 flex items-center justify-center">
                    <span className="text-white/60 text-sm">image of interface</span>
                </div>
            </div>
        </section>
    );
}