import React from "react";
import { Upload, Cog, Sparkles, Download, MoveDown } from "lucide-react";

interface Step {
  title: string;
  text: string;
  icon: React.ElementType;
}

const steps: Step[] = [
  {
    title: "Upload a CSV File With the data",
    text: "User uploads a CSV containing candidate parameters (e.g., period, duration, transit depth, radii). The Front-end part sends it to the Backend.",
    icon: Upload,
  },
  {
    title: "Process the data",
    text: "The service cleans missing values, normalizes features, and unifies formats. It also drops noisy or uninformative columns to improve accuracy.",
    icon: Cog,
  },
  {
    title: "Make Prediction",
    text: "Our ML model estimates the probability that each object is a real exoplanet. Then writes the answers to a new CSV file and returns it back.",
    icon: Sparkles,
  },
  {
    title: "Download The File",
    text: "Download the new processed CSV file with the answers for every entry.",
    icon: Download,
  },
];

const HowItWorks: React.FC = () => {
  return (
    <section id="how-works" className="w-full bg-[#F7F2FF] py-16 md:py-24">
      <div className="max-w-6xl mx-auto px-4 md:px-8">
        {/* Title */}
        <h2 className="text-center text-3xl md:text-4xl font-semibold mb-14">
          <span className="text-purple-700 underline underline-offset-4 decoration-4">
            How
          </span>{" "}
          does it work?
        </h2>

        <div className="flex flex-col gap-10 relative w-3/4 mx-auto">
          {steps.map((step, index) => (
            <div key={index} className="flex items-start md:items-center gap-6 relative">
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 flex items-center justify-center rounded-md bg-purple-700 shadow-md">
                  <step.icon className="text-white w-8 h-8" />
                </div>
                {index < steps.length - 1 && (
                  <div className="h-5  relative">
                    <span className="absolute left-1/2 top-full -translate-x-1/2 text-gray-500 text-xl">
                      <MoveDown />
                    </span>
                  </div>
                )}
              </div>

              <div>
                <h3 className="font-semibold text-lg mb-2 text-gray-900">
                  {step.title}
                </h3>
                <p className="text-gray-700 leading-relaxed max-w-2xl">
                  {step.text}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
