export default function Footer() {
  return (
    <footer className="scroll-fade fade-target bg-[#1E6B70] py-6 px-6 mt-auto">
      <p className="text-white text-sm font-medium mb-1">
        made by ddmal etc etc sponsored by quebec etc ec
      </p>
      <p className="text-white text-sm">
        <a
          href="#"
          onClick={(e) => e.preventDefault()}
          className="hover:opacity-70 transition-opacity"
        >
          contact us!
        </a>
      </p>
    </footer>
  );
}