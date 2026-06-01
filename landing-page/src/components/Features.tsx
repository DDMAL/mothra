const FEATURES = ['feature 1', 'feature 2', 'feature 3', 'feature 4'];

export default function Features() {
    return (
    <section className="bg-[#4AADAA] py-16 px-6">
      <div className="max-w-4xl mx-auto">
        <h2 className="scroll-fade fade-target text-3xl font-bold italic text-center text-white mb-10">
          what mothra can do
        </h2>
        <div className="scroll-fade fade-target border-2 border-white rounded-3xl p-8">
          <div className="flex justify-around gap-4">
            {FEATURES.map((label) => (
              <div
                key={label}
                className="w-36 h-36 rounded-full bg-[#1E6B70] flex items-center justify-center text-center text-white text-sm px-4"
              >
                {label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}