import React from "react";
import { UploadCloud, Loader2, FileText, Download } from "lucide-react";

type Status = "idle" | "processing" | "ready";

const TryItYourself: React.FC = () => {
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const [file, setFile] = React.useState<File | null>(null);
  const [status, setStatus] = React.useState<Status>("idle");
  const [downloadUrl, setDownloadUrl] = React.useState<string | null>(null);

  const handleClick = () => {
    if (status === "idle") fileInputRef.current?.click();
  };

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const f = files[0];

    if (f.type !== "text/csv" && !f.name.toLowerCase().endsWith(".csv")) {
      alert("Please upload a CSV file only.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setFile(f);
    setStatus("processing");
    void processFile(f);
  };

  const processFile = async (f: File) => {
    try {
      const formData = new FormData();
      formData.append("file", f);

      const response = await fetch("https://api.chiamo.ch/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process file");
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
      setStatus("ready");
    } catch (error) {
      console.error(error);
      setStatus("idle");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      alert("Processing failed. Please try again.");
    }
  };

  const onDragOver: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  };
  const onDrop: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  return (
    <section className="w-full bg-[#F7F2FF] py-16 md:py-24">
      <div className="max-w-6xl mx-auto px-4 text-center">
        <h2 className="text-3xl md:text-4xl font-semibold mb-12">
          <span className="text-purple-700 underline underline-offset-4 decoration-4">
            Try
          </span>{" "}
          it to yourself
        </h2>

        {status === "idle" && (
          <div
            onClick={handleClick}
            onDragOver={onDragOver}
            onDrop={onDrop}
            className="w-72 h-72 mx-auto flex flex-col items-center justify-center rounded-xl border-4 border-dotted border-purple-700 cursor-pointer transition hover:bg-purple-100/40"
          >
            <div className="w-20 h-20 bg-purple-700 flex items-center justify-center rounded-md mb-4">
              <UploadCloud className="text-white w-15 h-15" />
            </div>
            <p className="text-gray-800">Click</p>
            <p className="text-gray-500 text-sm">or</p>
            <p className="text-gray-800">Drag and Drop</p>

            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>
        )}

        {status === "processing" && (
          <div className="w-72 h-72 mx-auto flex flex-col items-center justify-center rounded-xl border-4 border-dotted border-purple-700 transition">
            <div className="w-20 h-20 bg-purple-700 flex items-center justify-center rounded-md mb-4">
              <Loader2 className="text-white w-10 h-10 animate-spin" />
            </div>
            <p className="text-gray-800 font-medium">Processing data...</p>
          </div>
        )}

        {status === "ready" && (
          <div className="w-72 h-72 mx-auto flex flex-col items-center justify-center rounded-xl border-4 border-dotted border-purple-700 transition hover:bg-purple-100/40">
            <div className="w-20 h-20 bg-purple-700 flex items-center justify-center rounded-md mb-4">
              <FileText className="text-white w-10 h-10" />
            </div>
            {file && (
              <p className="text-gray-700 text-sm mb-3 max-w-[15rem] truncate px-4">
                {file.name}
              </p>
            )}
            <a
              href={downloadUrl ?? "#"}
              download="processed.csv"
              onClick={(e) => {
                if (!downloadUrl) e.preventDefault();
                setStatus("idle");
              }}
              className="px-5 py-2 rounded-full bg-purple-700 text-white font-medium shadow hover:brightness-110 transition"
            >
              <Download className="inline-block mr-2 w-4 h-4" />
              Download
            </a>
          </div>
        )}
      </div>
    </section>
  );
};

export default TryItYourself;
