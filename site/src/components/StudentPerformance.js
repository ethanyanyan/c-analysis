// site/src/components/StudentPerformance.js
import React, { useEffect, useState } from "react";
import Papa from "papaparse";
import { INSTRUMENTATION, NUMERIC_COLS } from "../constants";
import "./StudentPerformance.css";

export default function StudentPerformance() {
  const [numericStats, setNumericStats] = useState(null);
  const [categoricalCounts, setCategoricalCounts] = useState(null);

  useEffect(() => {
    Papa.parse(`${process.env.PUBLIC_URL}/data/StudentPerformanceFactors.csv`, {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: ({ data }) => {
        const clean = data.filter((r) => r.Exam_Score != null);
        // console.log(clean);
        setNumericStats(computeNumericStats(clean));
        setCategoricalCounts(computeCategoricalCounts(clean));
      },
      error: (err, file, inputElem, reason) => {
        console.error("Failed to parse CSV:", reason);
      },
    });
  }, []);

  function computeNumericStats(data) {
    const stats = {};
    NUMERIC_COLS.forEach((col) => {
      const vals = data
        .map((r) => r[col])
        .filter((v) => typeof v === "number" && !isNaN(v))
        .sort((a, b) => a - b);

      if (!vals.length) return;
      const n = vals.length;
      const sum = vals.reduce((a, b) => a + b, 0);
      const mean = sum / n;
      const median = (vals[((n - 1) / 2) | 0] + vals[(n / 2) | 0]) / 2;
      const q1 = vals[Math.floor((n + 1) / 4) - 1] ?? vals[0];
      const q3 = vals[Math.ceil((3 * (n + 1)) / 4) - 1] ?? vals[n - 1];
      const sd = Math.sqrt(
        vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)
      );

      stats[col] = {
        n,
        mean,
        median,
        min: vals[0],
        q1,
        q3,
        max: vals[n - 1],
        sd,
      };
    });
    return stats;
  }

  function computeCategoricalCounts(data) {
    const cats = {};
    INSTRUMENTATION.filter((c) => !NUMERIC_COLS.includes(c.name)).forEach(
      (c) => (cats[c.name] = {})
    );
    data.forEach((r) => {
      Object.keys(cats).forEach((col) => {
        const v = r[col] ?? "Missing";
        cats[col][v] = (cats[col][v] || 0) + 1;
      });
    });
    return cats;
  }

  return (
    <div className="student-perf">
      <h2>Student Performance Factors</h2>
      <p>
        Dataset URL:&nbsp;
        <a
          href="https://www.kaggle.com/datasets/lainguyn123/student-performance-factors"
          target="_blank"
          rel="noopener noreferrer"
        >
          Kaggle: Student Performance Factors
        </a>
      </p>

      <h3>1. Instrumentation</h3>
      <table>
        <thead>
          <tr>
            <th>Attribute</th>
            <th>Description</th>
            <th>Scale</th>
          </tr>
        </thead>
        <tbody>
          {INSTRUMENTATION.map((c) => (
            <tr key={c.name}>
              <td>{c.name}</td>
              <td>{c.desc}</td>
              <td>{c.scale}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>2. Numeric Descriptives</h3>
      {numericStats ? (
        <table>
          <thead>
            <tr>
              <th>Variable</th>
              <th>n</th>
              <th>Mean</th>
              <th>Median</th>
              <th>Min</th>
              <th>Q1</th>
              <th>Q3</th>
              <th>Max</th>
              <th>Std. Dev.</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(numericStats).map(([col, s]) => (
              <tr key={col}>
                <td>{col}</td>
                <td>{s.n}</td>
                <td>{s.mean.toFixed(2)}</td>
                <td>{s.median.toFixed(2)}</td>
                <td>{s.min}</td>
                <td>{s.q1.toFixed(2)}</td>
                <td>{s.q3.toFixed(2)}</td>
                <td>{s.max}</td>
                <td>{s.sd.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="loading">Loading numeric summaries…</p>
      )}

      <h3>3. Categorical Frequencies</h3>
      {categoricalCounts ? (
        Object.entries(categoricalCounts).map(([col, counts]) => (
          <div key={col}>
            <h4>{col}</h4>
            <table>
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(counts).map(([val, cnt]) => (
                  <tr key={val}>
                    <td>{val}</td>
                    <td>{cnt}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))
      ) : (
        <p className="loading">Loading categorical frequencies…</p>
      )}
    </div>
  );
}
