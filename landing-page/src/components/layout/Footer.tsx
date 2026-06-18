export default function Footer() {
  return (
    <footer className="scroll-fade fade-target bg-white border-t border-gray-200 py-8 px-6">
      <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-center gap-6 sm:gap-16">
        <img
          src="/schulich_logo.png"
          alt="Schulich School of Music, McGill University"
          className="h-10 object-contain"
        />
        <img
          src="/Ddmal_logo_transp-bg_no-border_1600w.png"
          alt="DDMAL - Distributed Digital Music Archives and Libraries Lab"
          className="h-7 object-contain"
        />
      </div>
    </footer>
  );
}
