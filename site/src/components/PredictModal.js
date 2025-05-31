// site/src/components/PredictModal.js
import React, { useState } from "react";
import Modal from "react-modal"; // or your modal library of choice

Modal.setAppElement("#root");

export default function PredictModal({ isOpen, onRequestClose }) {
  // list out all your predictors
  const fields = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Family_Income", // ordinal
    "Distance_from_Home", // ordinal
    "Motivation_Level", // ordinal
    "Parental_Involvement", // ordinal
    "Access_to_Resources", // ordinal
    "Physical_Activity",
    "Peer_Influence",
    "Internet_Access",
    "Learning_Disabilities",
    "Gender",
  ];
  const [form, setForm] = useState(
    Object.fromEntries(fields.map((f) => [f, ""]))
  );
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
    const { predicted_score } = await resp.json();
    setResult(predicted_score.toFixed(2));
    setLoading(false);
  };

  return (
    <Modal isOpen={isOpen} onRequestClose={onRequestClose}>
      <h2>Predict Exam Score</h2>
      <form onSubmit={handleSubmit}>
        {fields.map((f) => (
          <div key={f} style={{ marginBottom: 8 }}>
            <label>
              {f.replace(/_/g, " ")}:<br />
              <input
                name={f}
                value={form[f]}
                onChange={handleChange}
                required
              />
            </label>
          </div>
        ))}
        <button type="submit" disabled={loading}>
          {loading ? "Predicting…" : "Run Prediction"}
        </button>
      </form>

      {result !== null && (
        <p style={{ marginTop: 16 }}>
          <strong>Predicted Exam Score:</strong> {result}
        </p>
      )}

      <button onClick={onRequestClose} style={{ marginTop: 16 }}>
        Close
      </button>
    </Modal>
  );
}
