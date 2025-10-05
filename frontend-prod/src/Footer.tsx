import React from "react";
import { Mail} from "lucide-react";

type Member = {
  name: string;
  email: string;
};

const members: Member[] = [
  {
    name: "Dima Puntus",
    email: "dimapuntus1@gmail.com",
  },
  {
    name: "Danylo Maltsev",
    email: "danilvolodimirovich@gmail.com",
  },
  {
    name: "Ivan Puntus",
    email: "ivanpuntus1@gmail.com",
  },
];

const Footer: React.FC = () => {
  return (
    <footer className="bg-[#2A0854] text-white">
      <div className="h-1 w-full bg-gradient-to-r from-purple-300/40 via-white/40 to-purple-300/40" />

      <div className="max-w-6xl mx-auto px-4 py-10">
        <h3 className="text-center text-xl md:text-2xl font-semibold tracking-wide mb-8">
          Made by <span className="underline underline-offset-4 decoration-4">Cosmo Hunters</span>
        </h3>

        {/* cards */}
        <ul className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {members.map((m) => (
            <li
              key={m.name}
              className="rounded-xl bg-white/5 backdrop-blur-sm border border-white/10 p-4 hover:border-white/20 transition"
            >
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-full bg-white/15 flex items-center justify-center text-lg font-semibold">
                  {m.name
                    .split(" ")
                    .slice(0, 2)
                    .map((s) => s[0])
                    .join("")}
                </div>

                <div>
                  <p className="font-semibold leading-tight">{m.name}</p>
                </div>
              </div>

              <div className="mt-4 flex items-center gap-3 text-sm">
                <Mail className="w-4 h-4 opacity-80" />
                <a
                  href={`mailto:${m.email}`}
                  className="hover:underline break-all"
                >
                  {m.email}
                </a>
              </div>

            </li>
          ))}
        </ul>

        <div className="mt-10 text-center text-xs text-white/70">
          © {new Date().getFullYear()} Cosmo Hunters — NASA Space Apps Challenge
        </div>
      </div>
    </footer>
  );
};

export default Footer;
