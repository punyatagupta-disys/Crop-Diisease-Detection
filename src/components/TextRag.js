import React, { useState } from "react";
import axios from "axios";

function TextRAG() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:8000/rag", {
        question: question,
      });
      setAnswer(res.data.answer);
    } catch (err) {
      console.error("Error fetching answer:", err);
      setAnswer("Failed to fetch answer from backend.");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>🌱 Plant Disease Assistant (RAG)</h2>
      <input
        type="text"
        placeholder="Ask a question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: "60%", padding: "10px", marginRight: "10px" }}
      />
      <button onClick={handleSubmit}>Ask</button>
      <div style={{ marginTop: "20px" }}>
        <strong>Answer:</strong>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default TextRAG;
