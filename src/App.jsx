import { useState, useEffect } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import { Header } from "./components/Header";

function App() {
  const [logs, setLogs] = useState([]); // State to store logs
  const [selectedFile, setSelectedFile] = useState(null); // State for selected file

  // Function to fetch logs from backend
  const getArchives = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/list-logs");
      if (!response.ok) {
        throw new Error("Failed to fetch logs");
      }
      const data = await response.json();
      setLogs(data); // Store the logs in the state
    } catch (error) {
      console.error("Error fetching logs:", error);
    }
  };

  // Function to upload the selected file
  const uploadArchive = async () => {
    if (!selectedFile) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload-log", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload file");
      }

      // Clear selected file after successful upload
      setSelectedFile(null);
      document.getElementById("dropzone-file").value = "";

      // Refresh logs list to include the new file
      getArchives();
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  // Function to remove selected file
  const removeSelectedFile = () => {
    setSelectedFile(null);
    document.getElementById("dropzone-file").value = "";
  };

  // Load logs when component mounts
  useEffect(() => {
    getArchives();
  }, []);

  return (
    <div>
      <Header />
     
      <div className="flex flex-col gap-4">
        <div className="mt-4">
          <h2 className="text-lg font-semibold">Logs disponíveis:</h2>
          <ul className="flex flex-col gap-2">
            {logs.map((log, index) => (
              <li
                key={index}
                className="text-gray-700 dark:text-gray-300 border rounded-[4px] border-1 border-[#ededed] p-2 cursor-pointer hover:bg-gray-100"
              >
                {log}
              </li>
            ))}
          </ul>
        </div>
        {!selectedFile ? (
          <div className="flex items-center justify-center w-full">
            <label
              htmlFor="dropzone-file"
              className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600"
            >
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg
                  className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
                  aria-hidden="true"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 20 16"
                >
                  <path
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
                  />
                </svg>
                <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                  <span className="font-semibold">Click para escolher</span> ou
                  arraste até aqui
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  XES, BPMN, Petri
                </p>
              </div>
              <input
                id="dropzone-file"
                type="file"
                className="hidden"
                onChange={(e) => setSelectedFile(e.target.files[0])}
              />
            </label>
          </div>
        ) : (
          ""
        )}

        {/* Display selected file with an option to remove it */}
        {selectedFile && (
          <div className="mt-4 p-2 border rounded-lg bg-gray-50 dark:bg-gray-800 flex items-center justify-between">
            <span className="text-gray-700 dark:text-gray-300">
              {selectedFile.name}
            </span>
            <button
              onClick={removeSelectedFile}
              className="text-red-500 hover:text-red-700"
            >
              Remove
            </button>
          </div>
        )}

        <button
          className="mt-4 p-2 bg-blue-500 text-white rounded"
          onClick={uploadArchive}
        >
          Upload Selected File
        </button>
      </div>
    </div>
  );
}

export default App;
