// site/src/components/PredictModal.js
import React, { useState } from "react";

export default function PredictModal({ isOpen, onClose }) {
  const initial = {
    Hours_Studied: 0,
    Attendance: 0,
    Sleep_Hours: 0,
    Previous_Scores: 0,
    Tutoring_Sessions: 0,
    Family_Income_Num: 2, // e.g. Medium
    Distance_from_Home_Num: 2, // Moderate
    Motivation_Level_Num: 2,
    Parental_Involvement_Num: 2,
    Access_to_Resources_Num: 2,
    Physical_Activity: 0,
    Peer_Influence: "Neutral",
    Internet_Access: "Yes",
    Learning_Disabilities: "No",
    Gender: "Female",
  };
  const [form, setForm] = useState(initial);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: value }));
  };

  const predict = async () => {
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
    const json = await res.json();
    setResult(json.predicted_score.toFixed(1));
  };

  if (!isOpen) return null;
  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>Predict Exam Score</h2>
        {Object.entries(form).map(([key, val]) => (
          <div key={key} className="field">
            <label>{key}</label>
            <input name={key} value={val} onChange={handleChange} />
          </div>
        ))}
        <button onClick={predict}>Run Prediction</button>
        {result !== null && (
          <p>
            Predicted exam score: <strong>{result}</strong>
          </p>
        )}
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
