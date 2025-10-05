import React from "react";

const HeroSection: React.FC = () => {
  const scrollToNext = () => {
    const nextSection = document.getElementById("how-works");
    if (nextSection) {
      nextSection.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div
      className="relative h-screen bg-cover bg-center flex flex-col items-center justify-center text-white"
      style={{ backgroundImage: "url('/imgs/bg.png')" }}
    >
      <div className="absolute inset-0 bg-black/40" />

      <div className="relative text-center z-10">
        <p className="text-sm md:text-base text-gray-300 mb-2">
          2025 NASA Space Apps Challenge
        </p>
        <h1 className="text-4xl md:text-6xl font-semibold mb-8">
          Cosmo Hunters
        </h1>
        <button
          onClick={scrollToNext}
          className="px-8 py-3 cursor-pointer border-none outline-none bg-white text-black font-medium rounded-full shadow-lg hover:bg-gradient-to-r hover:from-white hover:to-purple-200 hover:shadow-xl transition"
        >
          Let’s Start
        </button>
      </div>
    </div>
  );
};

export default HeroSection;
